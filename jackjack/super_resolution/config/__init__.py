import os

if "KERAS_HOME" in os.environ:
    _keras_dir = os.environ.get("KERAS_HOME")
else:
    _keras_base_dir = os.path.expanduser("~")
    _keras_dir = os.path.join(_keras_base_dir, ".keras")

def keras_home():
    # Private accessor for the keras home location.
    return _keras_dir
