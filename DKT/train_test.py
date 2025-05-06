import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from sklearn import metrics
import argparse
from Utils.model_utils import ExperimentLogger
from Utils.data_loader import load_dataset
from model import DKT
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties


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
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--l2_weight", type=float, default=1e-6)
    parser.add_argument("--mode", type=str, default='train', choices=['train', 'predict'], 
                       help='train or predict mode')
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
    # 在训练结束后保存最佳模型
    model_save_path = f'../saved_models/dkt_{args.data_set}.pt'
    torch.save({
        'model_state_dict': logger.best_model_state,
        'args': args
    }, model_save_path)
    print(f"Best model saved to {model_save_path}")
    
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


def predict(args, q_seqs, r_seqs):
    """
    使用训练好的模型进行预测
    Args:
        args: 模型参数
        q_seqs: 问题序列 shape: (batch_size, seq_len)
        r_seqs: 回答序列 shape: (batch_size, seq_len)
    Returns:
        predictions: 预测结果
    """
    # 加载模型
    model_path = f'../saved_models/dkt_{args.data_set}.pt'
    print(f"\n正在加载模型: {model_path}")
    
    checkpoint = torch.load(model_path, weights_only=False)
    print("模型加载成功!")
    
    # 初始化模型
    model = DKT(args).to(args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("模型初始化完成，开始预测...")
    
    with torch.no_grad():
        # 准备输入数据
        q_seqs = torch.LongTensor(q_seqs).to(args.device)
        r_seqs = torch.LongTensor(r_seqs).to(args.device)
        next_q = q_seqs[:, 1:]  # 预测下一个问题的答案
        
        # 获取预测结果
        pred = model(q_seqs[:, :-1], r_seqs[:, :-1], next_q)
        
    return pred.cpu().numpy()


def setup_chinese_font():
    """设置中文字体"""
    try:
        # 尝试使用系统已安装的中文字体
        font_paths = [
            '/usr/share/fonts/truetype/arphic/uming.ttc',  # Ubuntu
            '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',  # Debian
            '/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc',  # Arch
        ]
        
        font_found = False
        for font_path in font_paths:
            try:
                font = FontProperties(fname=font_path)
                plt.rcParams['font.family'] = font.get_name()
                font_found = True
                break
            except:
                continue
        
        if not font_found:
            # 如果没有找到系统字体，使用matplotlib自带的DejaVu字体
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
            
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        return True
    except:
        print("警告：设置中文字体失败，图表中的中文可能无法正确显示")
        return False

if __name__ == '__main__':
    args = get_args()
    
    if args.mode == 'train':
        print("\n=== 开始训练模式 ===")
        train_model(args)
    elif args.mode == 'predict':
        print("\n=== 知识追踪预测报告 ===")
        data_loader = load_dataset(args)
        batch = next(iter(data_loader['test']))
        seq_lens, pad_data, pad_answer, pad_index, pad_label, timestamps, attempts, answer_times = batch
        
        # 在创建图表之前设置字体
        setup_chinese_font()
        
        num_students = min(5, len(seq_lens))
        for student_idx in range(num_students):
            print(f"\n\n=== 学生 {student_idx + 1} 的学习诊断报告 ===")
            
            # 获取当前学生的序列
            q_seqs = pad_data[student_idx:student_idx+1].cpu().numpy().tolist()
            r_seqs = pad_answer[student_idx:student_idx+1].cpu().numpy().tolist()
            seq_len = seq_lens[student_idx].item()
            
            # 截取有效长度的序列
            q_seqs[0] = q_seqs[0][:seq_len]
            r_seqs[0] = r_seqs[0][:seq_len]
            
            predictions = predict(args, q_seqs, r_seqs)
            
            # Calculate statistics first
            correct_count = sum(r_seqs[0])
            total_count = len(r_seqs[0])
            accuracy = correct_count/total_count
            
            # Calculate knowledge stats
            knowledge_stats = {}
            for i, q_id in enumerate(q_seqs[0]):
                if q_id not in knowledge_stats:
                    knowledge_stats[q_id] = {'count': 0, 'correct': 0}
                knowledge_stats[q_id]['count'] += 1
                if r_seqs[0][i] == 1:
                    knowledge_stats[q_id]['correct'] += 1
            
            # Now create plots
            plt.figure(figsize=(20, 12))
            
            # 1. 答题正确率饼图
            plt.subplot(231)
            labels = ['正确答题\n({} 题)'.format(correct_count), 
                     '错误答题\n({} 题)'.format(total_count - correct_count)]
            colors = ['#2ecc71', '#e74c3c']
            sizes = [correct_count, total_count - correct_count]  # 添加这行
            plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
            plt.title('学生答题正确率分布\n总题数: {} 题'.format(total_count), 
                     pad=20, fontsize=12)
            
            # 2. 知识点掌握情况条形图
            plt.subplot(232)
            knowledge_ids = list(knowledge_stats.keys())
            correct_rates = [stats['correct']/stats['count'] for stats in knowledge_stats.values()]
            y_pos = np.arange(len(knowledge_ids))
            bars = plt.barh(y_pos, correct_rates, color='#3498db')
            plt.yticks(y_pos, [f'知识点{kid}({knowledge_stats[kid]["count"]}题)' 
                              for kid in knowledge_ids])
            plt.xlabel('正确率')
            plt.title('各知识点掌握程度分析', pad=20, fontsize=12)
            
            # 为条形图添加数值标签
            for i, bar in enumerate(bars):
                width = bar.get_width()
                plt.text(width, bar.get_y() + bar.get_height()/2,
                        f'{width:.1%}', ha='left', va='center')
            
            # 3. 学习预测趋势图
            plt.subplot(233)
            x = range(1, len(predictions[0]) + 1)
            plt.plot(x, predictions[0], marker='o', label='预测正确率', color='#2980b9')
            plt.axhline(y=0.6, color='#e74c3c', linestyle='--', label='及格线(60%)')
            plt.fill_between(x, predictions[0], 0.6, 
                           where=(predictions[0] >= 0.6),
                           color='#2ecc71', alpha=0.3, label='优秀区间')
            plt.fill_between(x, predictions[0], 0.6,
                           where=(predictions[0] < 0.6),
                           color='#e74c3c', alpha=0.3, label='待提升区间')
            plt.xlabel('题目序号')
            plt.ylabel('预测正确率')
            plt.title('学习能力预测趋势', pad=20, fontsize=12)
            plt.legend(loc='best')
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # 4. 知识点练习分布图
            plt.subplot(234)
            counts = [stats['count'] for stats in knowledge_stats.values()]
            bars = plt.bar(knowledge_ids, counts, color='#9b59b6')
            plt.xlabel('知识点编号')
            plt.ylabel('练习题目数量')
            plt.title('知识点练习题目分布', pad=20, fontsize=12)
            
            # 为柱状图添加数值标签
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}题',
                        ha='center', va='bottom')
            
            # 5. 预测掌握度热力图
            plt.subplot(235)
            pred_matrix = np.zeros((len(knowledge_stats), 1))
            for i, q_id in enumerate(knowledge_stats.keys()):
                q_predictions = [pred for j, pred in enumerate(predictions[0]) 
                               if j < len(q_seqs[0])-1 and q_seqs[0][j+1] == q_id]
                if q_predictions:
                    pred_matrix[i, 0] = sum(q_predictions) / len(q_predictions)
            
            im = plt.imshow(pred_matrix, aspect='auto', cmap='RdYlGn')
            plt.colorbar(im, label='预测掌握度')
            plt.yticks(range(len(knowledge_ids)), 
                      [f'知识点{kid}({knowledge_stats[kid]["count"]}题)' 
                       for kid in knowledge_ids])
            plt.title('知识点预测掌握度分析\n(绿色表示掌握度高，红色表示需要加强)', 
                     pad=20, fontsize=12)
            
            # 添加总标题
            plt.suptitle(f'学生{student_idx + 1}知识掌握情况分析报告', 
                        fontsize=16, y=1.02)
            
            # 调整布局并保存
            plt.tight_layout()
            plt.savefig(f'student_{student_idx+1}_report.png', 
                       bbox_inches='tight', dpi=300)
            plt.close()
            
            print(f"\n=== 学生{student_idx + 1}学习诊断报告图表说明 ===")
            print("1. 答题正确率分布：展示学生整体答题的正确与错误比例")
            print("2. 知识点掌握程度：展示各知识点的正确率，条形越长表示掌握越好")
            print("3. 学习预测趋势：展示学生未来答题的预测正确率走势")
            print("4. 知识点练习分布：展示各知识点的练习题目数量")
            print("5. 预测掌握度分析：使用颜色深浅展示各知识点的预测掌握程度")
            print(f"\n报告图表已保存为: student_{student_idx+1}_report.png")
            print("\n一、当前学习情况")
            correct_count = sum(r_seqs[0])
            total_count = len(r_seqs[0])
            accuracy = correct_count/total_count
            print(f"1. 答题表现：")
            print(f"   - 已完成题目：{total_count}题")
            print(f"   - 答对题目：{correct_count}题")
            print(f"   - 当前正确率：{accuracy:.2%}")
            
            print("\n2. 知识掌握情况：")
            knowledge_stats = {}
            for i, q_id in enumerate(q_seqs[0]):
                if q_id not in knowledge_stats:
                    knowledge_stats[q_id] = {'count': 0, 'correct': 0}
                knowledge_stats[q_id]['count'] += 1
                if r_seqs[0][i] == 1:
                    knowledge_stats[q_id]['correct'] += 1
            
            for q_id, stats in knowledge_stats.items():
                correct_rate = stats['correct'] / stats['count']
                status = '优秀' if correct_rate >= 0.8 else '良好' if correct_rate >= 0.6 else '待加强'
                print(f"   知识点{q_id}：")
                print(f"   - 做题数：{stats['count']}题")
                print(f"   - 正确率：{correct_rate:.2%}")
                print(f"   - 掌握程度：{status}")
                
            print("\n二、学习预测分析")
            avg_pred = predictions[0].mean()
            print(f"1. 整体预测：")
            print(f"   - 知识掌握度：{avg_pred:.2%}")
            status = '优秀' if avg_pred >= 0.8 else '良好' if avg_pred >= 0.6 else '一般' if avg_pred >= 0.4 else '需要关注'
            print(f"   - 学习状态：{status}")
            
            print("\n2. 具体知识点预测：")
            for q_id in knowledge_stats.keys():
                q_predictions = [pred for i, pred in enumerate(predictions[0]) 
                               if i < len(q_seqs[0])-1 and q_seqs[0][i+1] == q_id]
                if q_predictions:
                    avg_q_pred = sum(q_predictions) / len(q_predictions)
                    status = '优秀' if avg_q_pred >= 0.8 else '良好' if avg_q_pred >= 0.6 else '需要关注'
                    print(f"   知识点{q_id}：")
                    print(f"   - 预测掌握度：{avg_q_pred:.2%}")
                    print(f"   - 预测状态：{status}")
            
            print("\n三、教学建议")
            print("1. 重点关注：")
            weak_points = [q_id for q_id, stats in knowledge_stats.items() 
                          if stats['correct']/stats['count'] < 0.6]
            if weak_points:
                print(f"   建议重点关注知识点：{', '.join(map(str, weak_points))}")
                print("   原因：这些知识点的正确率较低，需要加强练习")
            
            print("\n2. 学习建议：")
            if avg_pred < 0.6:
                print("   - 建议进行系统性知识梳理")
                print("   - 可以从基础知识点开始，逐步提高难度")
            elif avg_pred > 0.8:
                print("   - 当前学习状态优秀")
                print("   - 建议尝试更具挑战性的题目，以保持学习兴趣")
            else:
                print("   - 建议针对性练习薄弱知识点")
                print("   - 可以通过多样化的练习方式提高学习效果")