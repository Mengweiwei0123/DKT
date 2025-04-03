import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import os
import pandas as pd


def load_map(path, node1_name, node2_name):
    df = pd.read_csv(path)
    # ques2skill = torch.tensor(ques2skill_df['skill']).view(-1, 1)
    # 创建一个空字典，用于存储ques->skill的索引列表
    index_dict = {}
    # 遍历数据框的每一行
    for index, row in df.iterrows():
        node1 = row[node1_name]
        node2 = row[node2_name]
        # 检查字典中是否已经有ques的索引列表，如果没有，则创建一个空列表
        if node1 not in index_dict:
            index_dict[node1] = []
        # 将当前的skill添加到ques的索引列表中
        index_dict[node1].append(node2)
    # 找到最小和最大的ques值，然后生成连续的ques范围
    min_node1 = min(index_dict.keys())
    max_node1 = max(index_dict.keys())
    max_node2 = max(index_dict.values())
    all_ques_range = range(min_node1, max_node1 + 1)
    # 创建一个二维列表，用于存储索引列表
    index_list = [torch.tensor(index_dict.get(ques, [])) for ques in all_ques_range]
    index_list = pad_sequence(index_list, batch_first=True, padding_value=max_node2[0])
    return index_list

# 加载问题到技能的multi-hot编码
def load_problem_skill_mapping(args):
    file_path = os.path.join(args.data_path, args.data_set, 'graph', 'ques_skill.csv')
    problem_ids, skill_ids, mapping = [], [], {}

    with open(file_path) as f:
        data = f.read().strip().split('\n')[1:]
        for entry in data:
            problem, skill = entry.split(',')
            skill_id = str(int(skill) - 1)
            if problem in mapping:
                mapping[problem] += ',' + skill_id
            else:
                mapping[problem] = skill_id
            problem_ids.append(int(problem))
            skill_ids.append(int(skill_id))

    max_problem_id = max(problem_ids)
    max_skill_id = max(skill_ids)
    print(f"问题数: {max_problem_id + 1}")
    print(f"技能数: {max_skill_id + 1}")

    embeddings = [[0] * (max_skill_id + 1) for _ in range(max_problem_id + 1)]
    for problem, skills in mapping.items():
        for skill in skills.split(","):
            embeddings[int(problem)][int(skill)] = 1

    return mapping, torch.tensor(embeddings).to(args.device)


# 数据加载函数
def load_dataset(args):
    file_paths = {}
    data_lists = {}
    datasets = {}
    dataloaders = {}
    shuffle_options = {'train': True, 'test': False}
    global DEVICE
    DEVICE = torch.device(args.device)

    for split in ['train', 'test']:
        file_paths[split] = os.path.join(args.data_path, args.data_set, 'train_test', f'{split}_{args.input}.txt')
        data_lists[split] = parse_file(file_paths[split], args.min_seq_len, args.max_seq_len, args.data_set)
        datasets[split] = MyDataset(*data_lists[split])
        dataloaders[split] = DataLoader(datasets[split], batch_size=args.batch_size, collate_fn=batch_collate_fn, shuffle=shuffle_options[split])

    print('数据加载完成!')
    return dataloaders


# 文件解析函数
def parse_file(filename, min_seq_len, max_seq_len, data_set):
    def split_sequence(length):
        splits = []
        while length > 0:
            if length >= max_seq_len:
                splits.append(max_seq_len)
                length -= max_seq_len
            elif length >= min_seq_len:
                splits.append(length)
                length -= length
            else:
                length -= min_seq_len
        return len(splits), splits

    seq_lengths, question_ids, timestamps, attempts, answer_times, answers = [], [], [], [], [], []

    with open(filename) as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if i % 6 == 0:
            seq_len = int(line)
            if seq_len < min_seq_len:
                i += 6
                continue
            k_split, splits = split_sequence(seq_len)
            seq_lengths += splits
        else:
            array = [eval(e) for e in line.split(',')]
            for j in range(k_split):
                slice_range = slice(max_seq_len * j, max_seq_len * (j + 1))
                if i % 6 == 1:
                    question_ids.append(array[slice_range])
                elif i % 6 == 2:
                    timestamps.append(array[slice_range])
                elif i % 6 == 3:
                    attempts.append(array[slice_range])
                elif i % 6 == 4:
                    answer_times.append(array[slice_range])
                elif i % 6 == 5:
                    answers.append(array[slice_range])

        i += 1

    assert len(seq_lengths) == len(question_ids) == len(timestamps) == len(attempts) == len(answer_times) == len(answers)
    return seq_lengths, question_ids, timestamps, attempts, answer_times, answers

# 数据集类
class MyDataset(Dataset):
    def __init__(self, seq_lengths, question_ids, timestamps, attempts, answer_times, answers):
        self.seq_lengths = seq_lengths
        self.question_ids = question_ids
        self.timestamps = timestamps
        self.attempts = attempts
        self.answer_times = answer_times
        self.answers = answers

    def __len__(self):
        return len(self.seq_lengths)

    def __getitem__(self, idx):
        seq_len = self.seq_lengths[idx]
        ques_id = self.question_ids[idx]
        timestamp = self.timestamps[idx]
        attempt = self.attempts[idx]
        ans_time = self.answer_times[idx]
        answer = self.answers[idx]

        sample_len = torch.tensor([seq_len - 1], dtype=torch.long)
        sample_exercise = torch.tensor(ques_id[:-1], dtype=torch.long)
        sample_answer = torch.tensor(answer[:-1], dtype=torch.long)
        sample_next_exercise = torch.tensor(ques_id[1:], dtype=torch.long)
        sample_next_answer = torch.tensor(answer[1:], dtype=torch.float)
        sample_timestamp = torch.tensor(timestamp[:-1], dtype=torch.float)
        sample_attempt = torch.tensor(attempt[:-1], dtype=torch.float)
        sample_ans_time = torch.tensor(ans_time[:-1], dtype=torch.float)

        return [sample_len, sample_exercise, sample_answer, sample_next_exercise, sample_next_answer, sample_timestamp,
                sample_attempt, sample_ans_time]


# 批处理函数
def batch_collate_fn(batch):
    batch.sort(key=lambda x: x[0], reverse=True)
    seq_lens = torch.cat([x[0] for x in batch])
    questions = pad_sequence([x[1] for x in batch], batch_first=True)
    answers = pad_sequence([x[2] for x in batch], batch_first=True)
    next_questions = pad_sequence([x[3] for x in batch], batch_first=True)
    next_answers = pad_sequence([x[4] for x in batch], batch_first=True)
    timestamps = pad_sequence([x[5] for x in batch], batch_first=True)
    attempts = pad_sequence([x[6] for x in batch], batch_first=True)
    ans_times = pad_sequence([x[7] for x in batch], batch_first=True)

    device_batch = [seq_lens, questions, answers, next_questions, next_answers, timestamps, attempts, ans_times]
    return [tensor.to(DEVICE) for tensor in device_batch]
