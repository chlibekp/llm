from .cli import main

# The guard matters: tokenizer worker processes are spawned, and spawn re-imports
# the main module - without it every worker would run the CLI again.
if __name__ == "__main__":
    raise SystemExit(main())
