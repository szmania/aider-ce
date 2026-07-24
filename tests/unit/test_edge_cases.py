import unittest

import yaml

from cecli.helpers.nested import deep_merge


class TestEdgeCases(unittest.TestCase):
    def test_empty_arrays(self):
        dict1 = {"skills_paths": []}
        dict2 = {"skills_paths": ["./project-skills"]}
        expected = {"skills_paths": ["./project-skills"]}
        merged = deep_merge(dict1, dict2, deep_merge_arrays=True)
        self.assertEqual(merged, expected)

    def test_null_and_missing_values(self):
        dict1 = {"skills_paths": ["./project-skills"]}
        dict2 = {"skills_paths": None}
        expected = {"skills_paths": None}
        merged = deep_merge(dict1, dict2, deep_merge_arrays=True)
        self.assertEqual(merged, expected)

        dict1 = {"skills_paths": ["./project-skills"]}
        dict2 = {}
        expected = {"skills_paths": ["./project-skills"]}
        merged = deep_merge(dict1, dict2, deep_merge_arrays=True)
        self.assertEqual(merged, expected)

    def test_non_array_fields_unchanged(self):
        dict1 = {"model": "gpt-4", "temperature": 0.5}
        dict2 = {"model": "claude-2", "dark_mode": True}
        expected = {"model": "claude-2", "temperature": 0.5, "dark_mode": True}
        merged = deep_merge(dict1, dict2, deep_merge_arrays=True)
        self.assertEqual(merged, expected)

    def test_malformed_yaml(self):
        """Verify graceful error handling with invalid YAML syntax."""
        malformed_yaml = "skills_paths: [unclosed quote\n"
        with self.assertRaises(yaml.YAMLError):
            yaml.safe_load(malformed_yaml)

        # Verify the try-except pattern works without crashing
        try:
            yaml.safe_load(malformed_yaml)
        except yaml.YAMLError:
            pass  # Expected — graceful handling confirmed


if __name__ == "__main__":
    unittest.main()
