from .treebank import TreeBank

def validate_conllu(*conllu_file_paths: str):
    errors: dict[str, AssertionError] = {}
    for conllu_file_path in conllu_file_paths:
        print(f"Validating {conllu_file_path}... ", end='', flush=True)
        try:
            TreeBank.from_conllu_file(conllu_file_path)
            print("Passed")
        except AssertionError as e:
            print("Failed")
            errors[conllu_file_path] = e
    if errors:
        print("\nReasons for Failure (only the first one encountered in each file):")
        for i, (conllu_file_path, error) in enumerate(errors.items(), start=1):
            print(f"{i}) {conllu_file_path}:\n{error}")
