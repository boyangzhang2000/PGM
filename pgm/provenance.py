"""Content-addressed source snapshots shared by search and benchmark runs."""
import hashlib
import zipfile


def snapshot_code(root, output):
    files = sorted(path for name in ('pgm', 'guided_diffusion', 'train') for path in (root/name).rglob('*.py'))
    files += sorted(path for name in ('models', 'operators', 'train')
                    for path in (root/'configs'/name).rglob('*.yaml'))
    contents = [(path.relative_to(root).as_posix(), path.read_bytes()) for path in files]
    digest = hashlib.sha256()
    for name, content in contents:
        digest.update(name.encode())
        digest.update(content)
    signature = digest.hexdigest()
    destination = output / f'source_{signature[:16]}.zip'
    if not destination.exists():
        with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED) as archive:
            for name, content in contents:
                archive.writestr(name, content)
    return signature
