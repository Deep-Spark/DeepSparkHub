build:
	@python setup.py build
	@mkdir -p _ext
	@mv $$(ls build/lib*/cuda_impl/*.so) _ext
	@rm -rf build

install:
	@python setup.py install
	@rm -rf build
	@rm -rf dist
	@rm -rf cuda_impl.egg-info
