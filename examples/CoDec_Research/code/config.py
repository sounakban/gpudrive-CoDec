local_config = {
                'dataset_path': 'data/processed/construal/Settest/',     # Path to scenario files
                'num_parallel_envs': 2,
                'total_envs': 4,
                'device': "'cpu'",
                'sample_size_utility': 1,                                # Number of samples to compute expected utility of a construal
                'construal_count_baseline': 2,                            # Number of construals to sample for baseline data generation
                'trajectory_count_baseline': 1,                           # Number of baseline trajectories to generate per construal
                }

server_config = {
                'dataset_path': 'data/processed/construal/Set2/',        # Path to scenario files
                'num_parallel_envs': 25,
                'total_envs': 25,
                'device': "'cuda' if torch.cuda.is_available() else 'cpu'",
                'sample_size_utility': 40,                                # Number of samples to compute expected utility of a construal
                'construal_count_baseline': 2,                            # Number of construals to sample for baseline data generation
                'trajectory_count_baseline': 3,                           # Number of baseline trajectories to generate per construal
                }