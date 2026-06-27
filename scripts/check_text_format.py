import argparse
from pathlib import Path


DEFAULT_FILES = [
    '.gitattributes',
    '.editorconfig',
    'basicsr/archs/wavelet_vssm_modules.py',
    'basicsr/archs/DeflareMamba_arch.py',
    'scripts/test_bs_wss_forward.py',
    'scripts/check_text_format.py',
    'scripts/verify_v2_5.py',
    'scripts/verify_v2_6.py',
    'options/DeflareMamba_flare7kpp_baseline_option.yml',
    'options/DeflareMamba_flare7kpp_bs_wss_option.yml',
    'README.md',
]

MIN_LINE_COUNTS = {
    'basicsr/archs/wavelet_vssm_modules.py': 100,
    'basicsr/archs/DeflareMamba_arch.py': 500,
    'scripts/test_bs_wss_forward.py': 50,
    'scripts/check_text_format.py': 40,
    'scripts/verify_v2_6.py': 50,
    'options/DeflareMamba_flare7kpp_bs_wss_option.yml': 100,
}


def check_file(root, name):
    path = root / name
    data = path.read_bytes()
    if b'\x00' in data:
        raise AssertionError(f'{path}: contains NUL bytes')
    if b'\r' in data:
        raise AssertionError(f'{path}: contains CR characters; expected LF-only line endings')
    line_count = data.count(b'\n')
    if line_count == 0:
        raise AssertionError(f'{path}: appears to be a single-line file')
    lines = len(data.decode('utf-8').splitlines())
    minimum = MIN_LINE_COUNTS.get(name)
    if minimum is not None and lines <= minimum:
        raise AssertionError(f'{path}: has {lines} lines, expected > {minimum}')
    print(f'{path}: ok, lines={lines}')


def main():
    parser = argparse.ArgumentParser(description='Check text files for sane line endings.')
    parser.add_argument('files', nargs='*', default=DEFAULT_FILES)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    for name in args.files:
        check_file(root, name)


if __name__ == '__main__':
    main()
