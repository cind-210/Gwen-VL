import json

input_path = "pretrain_t2t_mini.jsonl"
output_path = "pretrain_t2t_mini_100k.jsonl"

with open(input_path, "r", encoding="utf-8") as fin, \
     open(output_path, "w", encoding="utf-8") as fout:
    for i, line in enumerate(fin):
        if i >= 100000:
            break
        # 可选：验证 JSON 格式
        # data = json.loads(line)
        # fout.write(json.dumps(data, ensure_ascii=False) + "\n")
        fout.write(line)