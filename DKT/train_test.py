import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from sklearn import metrics
import argparse
from Utils.model_utils import ExperimentLogger
from Utils.data_loader import load_dataset
from model import DKT


def get_args():
    dataset_name = 'JunYi'

    question_numbers = {
        'ASSIST09': 15550,
        'ASSIST17': 1146,
        'JunYi': 662,
        'Statics': 633,
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--min_seq_len", type=int, default=3)
    parser.add_argument("--max_seq_len", type=int, default=200)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument("--input", type=str, default='all_feature')
    parser.add_argument("--data_path", type=str, default=r"../Datasets")
    parser.add_argument("--data_set", type=str, default=dataset_name)
    parser.add_argument("--ques_num", type=int, default=question_numbers[dataset_name])
    parser.add_argument('--remark', type=str, default='')
    parser.add_argument("--patience", type=int, default=15)

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--lr", type=int, default=0.001)
    parser.add_argument("--l2_weight", type=int, default=1e-6)

    # Setting for DKVMN
    parser.add_argument('--q_embed_dim', type=int, default=64, help='question embedding dimensions')
    parser.add_argument('--qa_embed_dim', type=int, default=128, help='answer and question embedding dimensions')
    parser.add_argument('--memory_size', type=int, default=64, help='memory size')
    parser.add_argument('--final_fc_dim', type=float, default=64, help='hidden state dim for final fc layer')

    # Setting for SAKT
    parser.add_argument('--num_attn_layer', type=int, default=4)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--encode_pos', action='store_true')
    parser.add_argument('--max_pos', type=int, default=1000)
    parser.add_argument('--drop_prob', type=float, default=0.20)

    # Setting for NTNKT
    parser.add_argument("--layers", type=int, default=3)

    # Setting for GLNC
    parser.add_argument("--ratio", type=float, default=0.3)

    return parser.parse_args()


def train_model(args):
    print("Configuration parameters:\n\n", args, "\n")
    logger = ExperimentLogger(args)
    data_loader = load_dataset(args)
    model = DKT(args).to(args.device)

    print(model)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.l2_weight)
    loss_function = nn.BCELoss()

    for epoch in range(args.epochs):
        logger.increment_epoch()
        total_loss = 0
        for batch in data_loader['train']:
            seq_lens, pad_data, pad_answer, pad_index, pad_label, timestamps, attempts, answer_times = batch
            predictions = model(pad_data, pad_answer, pad_index)
            packed_predictions = pack_padded_sequence(predictions, seq_lens.cpu(), enforce_sorted=True, batch_first=True)
            packed_labels = pack_padded_sequence(pad_label, seq_lens.cpu(), enforce_sorted=True, batch_first=True)

            loss = loss_function(packed_predictions.data, packed_labels.data)
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_metrics = evaluate_model(args, model, data_loader['train'])
        test_metrics = evaluate_model(args, model, data_loader['test'])
        logger.log_epoch(epoch, train_metrics, test_metrics, model)
        if logger.should_stop_early():
            break
        print("Total Loss", total_loss)
    logger.finalize_run(args)


def evaluate_model(args, model, data_loader):
    model.eval()
    true_labels, predicted_scores = [], []

    for batch in data_loader:
        seq_lens, pad_data, pad_answer, pad_index, pad_label, timestamps, attempts, answer_times = batch
        predictions = model(pad_data, pad_answer, pad_index)
        packed_predictions = pack_padded_sequence(predictions, seq_lens.cpu(), enforce_sorted=True, batch_first=True)
        packed_labels = pack_padded_sequence(pad_label, seq_lens.cpu(), enforce_sorted=True, batch_first=True)

        true_labels.append(packed_labels.data.cpu().contiguous().view(-1).detach())
        predicted_scores.append(packed_predictions.data.cpu().contiguous().view(-1).detach())

    all_predictions = torch.cat(predicted_scores, 0)
    all_true_labels = torch.cat(true_labels, 0)
    auc_score = metrics.roc_auc_score(all_true_labels, all_predictions)

    all_predictions[all_predictions >= 0.5] = 1.0
    all_predictions[all_predictions < 0.5] = 0.0
    accuracy = metrics.accuracy_score(all_true_labels, all_predictions)
    rmse = torch.sqrt(torch.mean((all_true_labels - all_predictions) ** 2)).item()

    model.train()
    return {'auc': auc_score, 'acc': accuracy, 'rmse': rmse}


if __name__ == '__main__':
    train_model(get_args())
