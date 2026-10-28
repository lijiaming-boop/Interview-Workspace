# Offline dependencies

## OpenVLA dlimp fork

- Archive: `dlimp_openvla-040105d.tar.gz`
- Upstream: `https://github.com/moojink/dlimp_openvla`
- Commit: `040105d256bd28866cc6620621a3d5f7b6b91b46`
- SHA256: `5a947924ce3cd4901de2362dd0e4dd8ef9f9c22b422f94da483971bd8b87004b`

Install without resolving dependencies:

```bash
python -m pip install --no-deps vendor/dlimp_openvla-040105d.tar.gz
```

The package expects TensorFlow 2.15 and TensorFlow Datasets 4.9.x. Install
those separately from an accessible Python package mirror when they are not
already present.
