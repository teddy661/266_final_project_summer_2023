for file in $(find ./results/scoring_dicts/ -type f); do python utils/evaluate_squadv2.py "./data/squadv2/dev-v2.0.json" "$file" "-o" "${file//scoring_dict_/}"; done
