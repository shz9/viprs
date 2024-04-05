.PHONY: build build-inplace test test-inplace dist redist install install-from-source clean uninstall publish-test publish

build:
	python3 setup.py build

build-inplace:
	python3 setup.py build_ext --inplace

test-inplace:
	PYTHONPATH=. pytest

test:
	python -m pytest

dist:
	python setup.py sdist bdist_wheel

redist: clean dist

install:
	python -m pip install .

install-from-source: dist
	python -m pip install dist/viprs-*.tar.gz

clean:
	$(RM) -r build dist *.egg-info
	$(RM) -r viprs/model/vi/*.c viprs/model/vi/*.cpp
	$(RM) -r viprs/utils/*.c viprs/utils/*.cpp
	$(RM) -r viprs/model/vi/*.so viprs/utils/*.so
	$(RM) -r .pytest_cache .tox temp output
	find . -name __pycache__ -exec rm -r {} +

uninstall:
	python -m pip uninstall viprs

publish-test:
	python -m twine upload -r testpypi dist/* --verbose

publish:
	python -m twine upload dist/* --verbose
