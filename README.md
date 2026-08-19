## EDM MDE Validation Tests

Set of `pytest` scripts to validate EDM or MDE implementations against [pyEDM](https://pypi.org/project/pyEDM/) and [dimx](https://pypi.org/project/dimx/) and [edmkit](https://pypi.org/project/edmkit/) implementations. 

## Getting Started

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) to manage dependencies and run the tests. You can install it with the following command:
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Synchronize the dependencies with the following command:
```
uv sync
```

Run the tests with the following command:
```
uv run pytest
```
