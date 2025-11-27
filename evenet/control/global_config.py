# config.py

import yaml
from copy import deepcopy
from pathlib import Path
from rich import get_console, box
from rich.align import Align
from rich.table import Table
from rich.tree import Tree
from rich.text import Text
from rich.console import Console

from evenet.control.event_info import EventInfo


class DotDict(dict):
    """Recursive dict with attribute-style access."""

    def __init__(self, d=None):
        super().__init__()
        d = d or {}
        for k, v in d.items():
            self[k] = self._wrap(v)

    def __getattr__(self, key):
        if key in self:
            return self[key]
        raise AttributeError(f"'DotDict' has no attribute '{key}'")

    def __setattr__(self, key, value):
        self[key] = self._wrap(value)

    def __delattr__(self, key):
        if key in self:
            del self[key]
        else:
            raise AttributeError(f"'DotDict' has no attribute '{key}'")

    def __deepcopy__(self, memo):
        copied = DotDict()
        for k, v in self.items():
            copied[k] = deepcopy(v, memo)
        return copied

    def _wrap(self, value):
        if isinstance(value, dict):
            return DotDict(value)
        elif isinstance(value, list):
            return [self._wrap(v) for v in value]
        return value

    def to_dict(self):
        result = {}
        for k, v in self.items():
            if isinstance(v, DotDict):
                result[k] = v.to_dict()
            elif isinstance(v, list):
                result[k] = [vv.to_dict() if isinstance(vv, DotDict) else vv for vv in v]
            else:
                result[k] = v
        return result

    def merge(self, override: dict):
        for k, v in override.items():
            if k in self and isinstance(self[k], DotDict) and isinstance(v, dict):
                self[k].merge(v)  # recursive merge
            else:
                self[k] = self._wrap(v)


class Config:
    """Config manager with optional base YAML as defaults."""

    def __init__(self):
        self._global_config = DotDict()
        self._defaults = DotDict()
        self.loaded = False
        self.skip_keys = ["event_info", "resonance"]

    def load_yaml(self, path: str | Path, current_dir: Path = None):
        path = Path(path)
        with open(path, 'r') as f:
            data = yaml.safe_load(f) or {}

        for section, content in data.items():
            if isinstance(content, dict) and "default" in content:
                default_path = path.parent / content.pop("default")
                with open(default_path, 'r') as inc:
                    inc_data = yaml.safe_load(inc) or {}
                self._defaults[section] = DotDict(inc_data)
                # merged = {**inc_data, **content}
                # DotDict(inc_data).merge(content)
                self._global_config[section] = deepcopy(self._defaults[section])
                self._global_config[section].merge(content)
            else:
                self._global_config[section] = self._global_config.get(section, DotDict())
                if isinstance(content, dict):
                    self._global_config[section].merge(content)
                else:
                    self._global_config[section] = content


        required = self.skip_keys
        missing = [key for key in required if key not in self._global_config]
        if missing:
            raise ValueError(f"Missing required config section(s): {', '.join(missing)}")

        self._defaults.pop("event_info", None)
        self._defaults.pop("resonance", None)
        self._global_config["event_info"] = EventInfo.construct(
            config=self._global_config.pop("event_info"),
            resonance_info=self._global_config["resonance"],
        )

        if 'process_info' in self._global_config:
            self._global_config['process_info'].pop('EXCLUDE', None)

        # special deal with file paths
        def resolve_path(d, key):
            if key not in d or d[key] is None:
                return
            raw = Path(d[key]).expanduser()
            if current_dir is not None:
                d[key] = (Path(current_dir) / raw).resolve()
            else:
                d[key] = raw.resolve()

        # Mapping of dict → list of keys
        path_fields = {
            'platform': ['data_parquet_dir', 'data_parquet_val_dir'],
            'logger': ['save_dir'],
            'options.Training': [
                'model_checkpoint_save_path',
                'model_checkpoint_load_path',
                'pretrain_model_load_path'
            ],
            'options.Dataset': ['normalization_file'],
        }

        # Helper to walk nested dicts
        def get_nested(cfg, dotted_key):
            parts = dotted_key.split('.')
            for p in parts:
                cfg = cfg[p]
            return cfg

        # Apply the resolver
        for section, keys in path_fields.items():
            d = get_nested(self._global_config, section)
            for k in keys:
                resolve_path(d, k)

        self.loaded = True

    def update(self, data: dict):
        self._global_config.merge(data)

    def to_dict(self):
        return self._global_config.to_dict()

    def save(self, path: str | Path):
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f)

    def __getattr__(self, key):
        if key == "_global_config":
            return object.__getattribute__(self, "_global_config")
        return getattr(self._global_config, key)

    def __getitem__(self, key):
        return self._global_config[key]

    def __str__(self):
        import pprint
        return pprint.pformat(self.to_dict(), indent=2)

    def _flatten_dict(self, d, skip_keys, parent_key="", ):
        items = []
        for k, v in d.items():
            if k in skip_keys:
                continue

            new_key = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, skip_keys, new_key).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def display(self, console=None):
        if console is None:
            console = Console()
        table = Table(
            title="🔧 Configuration Overview",
            # box=box.SQUARE,  # You can try box.ROUNDED or box.MINIMAL_DOUBLE_HEAD for variety
            header_style="bold magenta",
            # show_lines=False,  # no grid between rows
            # box=None,  # No borders on individual cells
            # expand=True
        )
        table.add_column("Parameter", style="cyan", no_wrap=False)
        table.add_column("Value", style="white", no_wrap=False)

        flat_default = self._flatten_dict(self._defaults, skip_keys=self.skip_keys)

        for section, content in self._global_config.items():
            if section in self.skip_keys:
                continue
            if isinstance(content, DotDict):
                # Add a bold separator row for the section
                separator_text = Text(f"──> {section.upper()} <──", style="bold white", justify="center")
                table.add_row(separator_text, "", end_section=True)

                flat_section = self._flatten_dict(content, parent_key=section, skip_keys=[])

                for key in sorted(flat_section.keys()):
                    val = flat_section[key]
                    default_val = flat_default.get(key, None)
                    style = "bold red" if val != default_val else "green"
                    table.add_row(key, str(val), style=style)

                table.add_row("", "", end_section=True)
            else:
                # For non-dict sections
                table.add_row(section, str(content), style="green")

        console.print(table)
        # Optional: print resonance tree
        if 'resonance' in self._global_config:
            tree = self.dict_to_rich_tree(self._global_config['resonance'])
            console.print(tree)

    def dict_to_rich_tree(self, data, tree=None):
        if tree is None:
            tree = Tree("Resonance Particles", guide_style="bold cyan")

        for key, value in data.items():
            if isinstance(value, dict):
                branch = tree.add(f"[bold yellow]{key}[/]")
                self.dict_to_rich_tree(value, branch)
            else:
                leaf = f"[green]{key}[/]: {value}"
                tree.add(leaf)
        return tree

    def to_logger(self):
        plain_dict = {}

        for key, value in self._global_config.items():
            if key in self.skip_keys:
                continue

            if isinstance(value, DotDict):
                plain_dict[key] = value.to_dict()
            else:
                plain_dict[key] = value

        return plain_dict


# --- Global instance --- #
global_config = Config()

if __name__ == '__main__':
    # Example usage
    # global_config.load_yaml("default.yaml")
    global_config.load_yaml("local_test.yaml")
    # global_config.load_yaml("preprocess_pretrain.yaml")
    global_config.display()

    a = 0

    # config.options.Network.hidden_dim
    # config.event_info.INPUTS.SEQUENTIAL.Source.mass
    # config.resonance.HadronicTop.'t/bqq'.Mass
