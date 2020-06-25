from functools import partial
from importlib import import_module
import collections
'''
Who's your favourite Metaphysician?
'''


def create_instance(fully_qualified_name, **kwargs):
	module_name, class_name = fully_qualified_name.rsplit('.', 1)
	module = import_module(module_name)

	class_ = getattr(module, class_name)
	instance = class_(**kwargs)

	return instance


def create_function(fully_qualified_name):
	if (isinstance(fully_qualified_name, collections.Callable)) :return fully_qualified_name

	if (fully_qualified_name.count('.') < 1): # Its a builtin!
		# Its a shame really do not use this, but I found sth more elegant
		#fn_ = getattr(globals()['__builtin__'], fully_qualified_name)
		fn_ = eval(fully_qualified_name)
	else:
		module_name, function_name = fully_qualified_name.rsplit('.', 1)
		module = import_module(module_name)

		fn_ = getattr(module, function_name)

	return fn_


def create_partial_function(fully_qualified_name, **kwargs):
	fn_ = create_function(fully_qualified_name=fully_qualified_name)

	if (len(kwargs) <= 0): return fn_ # Return if no kwargs have been passed

	fn_ = partial(fn_, **kwargs) # Partially apply kwargs

	return fn_


def prepare_invocation_on_obj(obj, function_name):
	fn_ = getattr(obj, function_name)

	return fn_


def getattr_from_module(fully_qualified_name, attr):
	module = import_module(fully_qualified_name)

	attr_ = getattr(module, attr)

	return attr_


def get_staticmethod_from_class(fully_qualified_name, static_method):
	cls_ = getattr_from_module(*fully_qualified_name.rsplit('.', 1))

	m_ = getattr(cls_, static_method)

	return m_
