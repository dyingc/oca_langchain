from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from string import Template
from typing import Any

from core.oauth2_token_manager import OCAOauth2TokenManager


DEFAULT_MODEL_INFO_URL = (
    "https://code-internal.aiservice.us-chicago-1.oci.oraclecloud.com/"
    "20250206/app/litellm/v1/model/info"
)
DEFAULT_OUTPUT_PATH = Path.home() / ".codex" / "config.toml"
DEFAULT_TEMPLATE_PATH = Path(__file__).resolve().parent.parent / "templates" / "codex_config.toml.template"
DEFAULT_MODEL_PREFERENCES = [
    "oca/gpt-5.3-codex",
    "oca/gpt-5-codex",
    "oca/gpt-5.4-pro",
    "oca/gpt-5.4",
    "oca/gpt-5",
    "oca/gpt-5.2",
    "oca/gpt-4.1",
]


@dataclass(frozen=True)
class ProfileSpec:
    profile_name: str
    model_id: str
    personality: str | None = None
    model_reasoning_effort: str | None = None


PROFILE_SPECS = [
    ProfileSpec("gpt-4-1", "oca/gpt-4.1"),
    ProfileSpec("gpt-5", "oca/gpt-5"),
    ProfileSpec("gpt-5-mini", "oca/gpt-5-mini"),
    ProfileSpec("gpt-5-1", "oca/gpt-5.1"),
    ProfileSpec("gpt-5-2", "oca/gpt-5.2"),
    ProfileSpec("gpt-5-codex", "oca/gpt-5-codex", personality="pragmatic"),
    ProfileSpec("gpt-5-1-codex", "oca/gpt-5.1-codex", personality="pragmatic"),
    ProfileSpec("gpt-5-1-codex-mini", "oca/gpt-5.1-codex-mini", personality="pragmatic"),
    ProfileSpec("gpt-5-1-codex-max", "oca/gpt-5.1-codex-max", personality="pragmatic"),
    ProfileSpec("gpt-5-2-codex", "oca/gpt-5.2-codex", personality="pragmatic"),
    ProfileSpec("gpt-5-3-codex", "oca/gpt-5.3-codex", personality="pragmatic", model_reasoning_effort="low"),
    ProfileSpec("gpt-5-4", "oca/gpt-5.4", personality="pragmatic"),
    ProfileSpec("gpt-5-4-pro", "oca/gpt-5.4-pro", personality="pragmatic"),
    ProfileSpec("gpt-5-4-mini", "oca/gpt-5.4-mini", personality="pragmatic"),
    ProfileSpec("gpt-5-4-nano", "oca/gpt-5.4-nano", personality="pragmatic"),
    ProfileSpec("openai-o3", "oca/openai-o3", personality="pragmatic"),
]


def build_token_manager(env_file: Path) -> OCAOauth2TokenManager:
    return OCAOauth2TokenManager(dotenv_path=str(env_file), debug=False)


def get_access_token(env_file: Path) -> str:
    token_manager = build_token_manager(env_file)
    return token_manager.get_access_token()


def fetch_model_info(
    token_manager: OCAOauth2TokenManager,
    access_token: str,
    model_info_url: str = DEFAULT_MODEL_INFO_URL,
) -> dict[str, Any]:
    response = token_manager.request(
        method="GET",
        url=model_info_url,
        headers={
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/json",
        },
        request_timeout=30,
    )
    return response.json()


def extract_model_catalog(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    catalog: dict[str, dict[str, Any]] = {}
    for item in payload.get("data", []):
        model_id = item.get("litellm_params", {}).get("model")
        if not model_id:
            continue
        model_info = item.get("model_info", {})
        catalog[model_id] = {
            "supported_api_list": list(model_info.get("supported_api_list") or []),
            "reasoning_effort_options": list(model_info.get("reasoning_effort_options") or []),
            "context_window": model_info.get("context_window"),
        }
    return catalog


def choose_default_model(catalog: dict[str, dict[str, Any]]) -> str:
    for candidate in DEFAULT_MODEL_PREFERENCES:
        if candidate in catalog:
            return candidate
    if not catalog:
        raise ValueError("No models returned from model info endpoint.")
    return sorted(catalog)[0]


def choose_default_profile(default_model: str) -> str:
    for spec in PROFILE_SPECS:
        if spec.model_id == default_model:
            return spec.profile_name
    return default_model.removeprefix("oca/").replace(".", "-").replace("_", "-").replace("/", "-")


def render_profile_block(spec: ProfileSpec) -> str:
    lines = [
        f"[profiles.{spec.profile_name}]",
        f'model = "{spec.model_id}"',
        'model_provider = "oca"',
        f'review_model = "{spec.model_id}"',
    ]
    if spec.personality:
        lines.append(f'personality = "{spec.personality}"')
    if spec.model_reasoning_effort:
        lines.append(f'model_reasoning_effort = "{spec.model_reasoning_effort}"')
    return "\n".join(lines)


def render_config(
    catalog: dict[str, dict[str, Any]],
    default_model: str,
    default_profile: str,
    template_path: Path = DEFAULT_TEMPLATE_PATH,
) -> str:
    available_profiles = [spec for spec in PROFILE_SPECS if spec.model_id in catalog]
    profiles_block = "\n\n".join(render_profile_block(spec) for spec in available_profiles)
    template = Template(template_path.read_text(encoding="utf-8"))
    return template.substitute(
        default_model=default_model,
        default_profile=default_profile,
        provider_default_model=default_model,
        profiles_block=profiles_block,
    ).rstrip() + "\n"


def backup_file(path: Path) -> Path | None:
    if not path.exists():
        return None
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_path = path.with_name(f"{path.name}.bak-{timestamp}")
    backup_path.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    return backup_path


def write_output(rendered_config: str, output_path: Path, backup: bool = True) -> Path | None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    backup_path = backup_file(output_path) if backup else None
    output_path.write_text(rendered_config, encoding="utf-8")
    return backup_path


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate ~/.codex/config.toml from the live OCA model catalog.")
    parser.add_argument("--env-file", default=".env", help="Path to the env file containing OAuth refresh configuration.")
    parser.add_argument(
        "--model-info-url",
        default=DEFAULT_MODEL_INFO_URL,
        help="Model info endpoint to query.",
    )
    parser.add_argument(
        "--template",
        default=str(DEFAULT_TEMPLATE_PATH),
        help="Path to the config template file.",
    )
    parser.add_argument(
        "--output",
        default="-",
        help='Output path. Use "-" to print the generated config to stdout.',
    )
    parser.add_argument(
        "--default-model",
        default="",
        help="Override the default model. Must exist in the fetched catalog.",
    )
    parser.add_argument(
        "--default-profile",
        default="",
        help="Override the default profile name in the generated config.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write directly to ~/.codex/config.toml and create a timestamped backup.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Disable backup creation when writing to a file.",
    )
    return parser


def main() -> int:
    parser = build_argument_parser()
    args = parser.parse_args()

    env_file = Path(args.env_file).expanduser()
    template_path = Path(args.template).expanduser()
    token_manager = build_token_manager(env_file)
    access_token = token_manager.get_access_token()
    catalog = extract_model_catalog(fetch_model_info(token_manager, access_token, args.model_info_url))

    default_model = args.default_model.strip() or choose_default_model(catalog)
    if default_model not in catalog:
        raise ValueError(f'Default model "{default_model}" is not present in the fetched model catalog.')

    default_profile = args.default_profile.strip() or choose_default_profile(default_model)
    rendered = render_config(
        catalog=catalog,
        default_model=default_model,
        default_profile=default_profile,
        template_path=template_path,
    )

    if args.apply:
        output_path = DEFAULT_OUTPUT_PATH
        backup_path = write_output(rendered, output_path, backup=not args.no_backup)
        print(f"Wrote {output_path}")
        if backup_path:
            print(f"Backup {backup_path}")
        return 0

    if args.output == "-":
        print(rendered, end="")
        return 0

    output_path = Path(args.output).expanduser()
    backup_path = write_output(rendered, output_path, backup=not args.no_backup)
    print(f"Wrote {output_path}")
    if backup_path:
        print(f"Backup {backup_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
