import unittest
import sys
from test_pipeline import TestPipeline

def run_tests():
    # Create a test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPipeline)
    
    # Create a test runner with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    
    # Run the tests
    result = runner.run(suite)
    
    # Return appropriate exit code
    return 0 if result.wasSuccessful() else 1

if __name__ == '__main__':
    sys.exit(run_tests()) 