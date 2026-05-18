import platform

#TODO: Placeholder-implementation
def get_hardware_info() -> dict:
    """Return basic hardware and Python runtime information.

        Returns:
            A dictionary containing CPU information, RAM information if available,
            and the Python version.
        """
    return {
        "cpu": platform.processor() or platform.machine(),
        "ram_gb": None,
        "python_version": platform.python_version(),
    }