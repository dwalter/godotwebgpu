def can_build(env, platform):
    return (env["opengl3"] or env["webgpu"]) and not env["disable_xr"]


def configure(env):
    pass


def get_doc_classes():
    return ["WebXRInterface"]


def get_doc_path():
    return "doc_classes"
