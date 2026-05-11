import platform

#TODO: Placeholder-implementation
def get_hardware_info() -> dict:
    return {
        "cpu": platform.processor() or platform.machine(),
        "ram_gb": None,
        "python_version": platform.python_version(),
    }