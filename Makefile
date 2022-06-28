.PHONY: build dist redist install install-from-source clean uninstall

build:
	python3 setup.py build

dist:
	python3 setup.py sdist bdist_wheel

redist: clean dist

install:
	pip install .

install-from-source: dist
	pip install dist/viprs-*.tar.gz

clean:
	$(RM) -r build dist *.egg-info
	$(RM) -r viprs/model/*.c viprs/utils/*.c
	find . -name __pycache__ -exec rm -r {} +

uninstall:
	pip uninstall viprs