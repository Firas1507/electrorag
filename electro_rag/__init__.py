"""ElectroRAG - Local AI assistant for electronics datasheets."""

__version__ = "1.0.0"
__author__ = "ElectroRAG Team"
__description__ = "A Retrieval-Augmented Generation system for querying electronics datasheets locally"
__url__ = "https://github.com/ElectroRAG/ElectroRAG"
__license__ = "MIT"

from src.rag import main

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python -m electro_rag \"Your question here\"")
        sys.exit(1)
    
    question = " ".join(sys.argv[1:])
    main()
