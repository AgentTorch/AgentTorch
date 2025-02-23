import vaex
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import math
import time


class AgentDataLoader:
    def __init__(self, base_dir=None, num_workers=8, batch_size=1000):
        """
        Initialize the AgentDataLoader.

        Parameters:
        base_dir (str): Base directory containing agent parquet files
        num_workers (int): Number of parallel workers for loading
        batch_size (int): Number of files to process in each batch
        """
        self.base_dir = base_dir
        self.num_workers = num_workers
        self.batch_size = batch_size

    def load_parquet_agents(self, file_path, columns=None):
        """
        Data loader for parquet files.

        Parameters:
        file_path (str): Path to the parquet file
        columns (list, optional): List of column names to load. If None, loads all columns.

        Returns:
        vaex.DataFrame: DataFrame containing the agent data
        """
        try:
            df = vaex.open(file_path)

            if columns is not None:
                available_cols = set(df.column_names)
                valid_cols = [col for col in columns if col in available_cols]
                if not valid_cols:
                    raise ValueError(
                        f"None of the requested columns {columns} found in the file"
                    )
                df = df[valid_cols]

            return df

        except Exception as e:
            print(f"Error loading parquet file {file_path}: {e}")
            return None

    def process_file_batch(self, file_paths, columns=None):
        """Helper function to process a batch of files"""
        dfs = []
        for file_path in file_paths:
            try:
                df = self.load_parquet_agents(str(file_path), columns)
                if df is not None:
                    dfs.append(df)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        return dfs

    def load_agent_files(self, base_dir=None, columns=None):
        """
        Load and process multiple parquet files in parallel.

        Parameters:
        base_dir (str): Base directory containing agent parquet files
        columns (list, optional): List of column names to load

        Returns:
        vaex.DataFrame: Concatenated DataFrame containing all agent data
        """
        if base_dir is not None:
            self.base_dir = base_dir

        if self.base_dir is None:
            raise ValueError(
                "base_dir must be specified either in __init__ or load_agent_files"
            )

        all_files = list(Path(self.base_dir).rglob("*.parquet"))
        total_files = len(all_files)
        print(f"Found {total_files} parquet files")

        num_batches = math.ceil(total_files / self.batch_size)
        all_dfs = []

        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            for batch_idx in range(num_batches):
                start_idx = batch_idx * self.batch_size
                end_idx = min((batch_idx + 1) * self.batch_size, total_files)
                batch_files = all_files[start_idx:end_idx]

                print(
                    f"Processing batch {batch_idx + 1}/{num_batches} ({start_idx} to {end_idx})"
                )

                chunk_size = math.ceil(len(batch_files) / self.num_workers)
                chunks = [
                    batch_files[i : i + chunk_size]
                    for i in range(0, len(batch_files), chunk_size)
                ]

                futures = [
                    executor.submit(self.process_file_batch, chunk, columns)
                    for chunk in chunks
                ]

                for future in futures:
                    batch_dfs = future.result()
                    all_dfs.extend(batch_dfs)

                if len(all_dfs) > self.num_workers * 2:
                    print("Concatenating intermediate results...")
                    all_dfs = [vaex.concat(all_dfs)]

        print("Concatenating final results...")
        final_df = vaex.concat(all_dfs)

        print(f"Loaded {len(final_df)} total agents")
        return final_df


# start_time = time.time()

# df = load_agent_files("output_v2/population")

# total_time = time.time() - start_time
# print(f"Total processing time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")

# analyze_agents(df)

# df.export
