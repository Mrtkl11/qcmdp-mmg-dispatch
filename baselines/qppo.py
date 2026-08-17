try:
    from .runner import main
except ImportError:
    from baselines.runner import main


if __name__ == "__main__":
    main("qppo")
