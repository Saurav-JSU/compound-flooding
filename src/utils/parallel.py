# src/utils/parallel.py
import concurrent.futures
import time
from tqdm import tqdm

def process_in_parallel(items, process_func, max_workers=None, desc="Processing"):
    """
    Process a list of items in parallel.
    
    Args:
        items: List of items to process
        process_func: Function to apply to each item
        max_workers: Maximum number of worker processes (None uses all available)
        desc: Description for the progress bar
        
    Returns:
        List of results
    """
    start_time = time.time()
    results = []
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks and collect futures
        futures = [executor.submit(process_func, item) for item in items]
        
        # Process results as they complete with a progress bar
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=desc):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Error processing item: {e}")
    
    end_time = time.time()
    print(f"Parallel processing completed in {end_time - start_time:.2f} seconds")
    
    return results