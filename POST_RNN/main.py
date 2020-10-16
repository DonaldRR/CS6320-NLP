import os
import argparse
import numpy as np
from tqdm import tqdm

import tensorboardX
import torch
from torch.utils.data import Dataset, DataLoader

from criterion import CELoss
from model import POSTModel
from dataset import POSTDataset
from utils import *

class LR_Scheduler:

    def __init__(self, init_lr, max_iters):
        self.init_lr = init_lr
        self.max_iters = max_iters
        self.t = 0

    def step(self):

        lr = self.init_lr * (1 - self.t / self.max_iters)**0.9
        self.t += 1

        return lr



def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, default='train')
    parser.add_argument('--embd', type=str, default='pretrained/glove.6B.100d.txt')
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--ckpt', type=str, default='./log/epoch_100.pth')
    parser.add_argument('--eval', action='store_true', default=False)
    args = parser.parse_args()

    return args


def train(args, model, dataset, dataloader, optim, criterion, tb_logger):
    
    global_step = 0
    with tqdm(total=args.epoch) as pbar:

        confusion_matrix = np.zeros((dataset.n_class, dataset.n_class))
        for i_epoch in range(args.epoch):

            model.train()
            loss_sum = torch.tensor(0).cuda().float()
            for input, gt in tqdm(dataloader):
                input = input.cuda()
                gt = gt.cuda()

                pred = model(input)
                loss = criterion(pred, gt)
                optim.zero_grad()
                # optim.lr = lr_scheduler.step()
                loss.backward()
                optim.step()

                tb_logger.add_scalar('train/loss', loss.detach().cpu().numpy(), global_step=global_step)
                loss_sum += loss

                pred = pred.detach().cpu().numpy()
                gt = gt.detach().cpu().numpy()
                for i_item in range(len(pred)):
                    confusion_matrix += compute_confusion_matrix(pred[i_item], gt[i_item], train_dataset.n_class)

                global_step += 1

            sample_sentences, sample_inputs, sample_tags = dataset.sample(5)
            model.eval()
            sample_preds = model(sample_inputs.cuda())
            sample_preds = sample_preds.detach().cpu().numpy()
            sample_preds = np.argmax(sample_preds, axis=2)
            sample_tags = sample_tags.numpy()
            for j in range(len(sample_preds)):
                tmp_len = (sample_tags[j] >= 0).sum()
                print('## Sample %d' % j)
                output_str = ""
                for k in range(tmp_len):
                    output_str += "%s/%s(%s) " % (
                        sample_sentences[j][k],
                        dataset.label_names[sample_tags[j][k]],
                        dataset.label_names[sample_preds[j][k]])
                print(output_str)


            torch.save({
                'model_state_dict': model.state_dict(),
                'optim_state_dict': optim.state_dict()
            }, os.path.join(args.logdir, 'epoch_%d.pth' % i_epoch))

            confusion_matrix[confusion_matrix == 0] = 1
            recalls = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=0)
            precisions = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=1)
            for k in range(train_dataset.n_class):
                tag_name = train_dataset.label_names[k]
                tb_logger.add_scalar('train/recall_%s' % tag_name, recalls[k], i_epoch)
                tb_logger.add_scalar('train/precision_%s' % tag_name, precisions[k], i_epoch)
            mean_recall = np.mean(recalls)
            mean_precision = np.mean(precisions)
            tb_logger.add_scalar('train/recall_mean', mean_recall, i_epoch)
            tb_logger.add_scalar('train/precision_mean', mean_precision, i_epoch)

            mean_loss = loss_sum.detach().cpu().numpy() / len(train_dataloader)
            msg = 'loss=%.4f m_recall=%.4f m_precision=%.4f' % (mean_loss, mean_recall, mean_precision)
            pbar.set_postfix_str(msg)
            pbar.update(1)


if __name__ == '__main__':

    args = parse_args()
    print('* Loading pretrained embeddings')
    pretrained_embeddings = load_pretrained_embeddings(args.embd)

    print('* Loading dataset')
    train_dataset = POSTDataset(args.train, indexes=pretrained_embeddings['word2idx'], max_len=200)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=4)
    print('* Loading model')
    model = POSTModel(pretrained_embeddings['embeddings'], train_dataset.n_class, hidden_dim=64)
    model.cuda()
    criterion = CELoss()
    #optim = torch.optim.SGD(model.parameters(),lr=0.02, momentum=0.2, nesterov=True, weight_decay=1e-5)
    optim = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    #lr_scheduler = LR_Scheduler(0.02, len(train_dataloader) * args.epoch)

    if not args.eval:
        print('* Initialize tensorboard logger')
        tb_logger = tensorboardX.SummaryWriter(args.logdir)
        print('* Start training')
        train(args, model, train_dataset, train_dataloader, optim, criterion, tb_logger)

    print('* Testing ')
    if not os.path.isfile(args.ckpt):
        print("! ERROR: Checkpoint %s is not available" % args.ckpt)
    else:
        print('## Loading checkpoint[%s] for testing' % args.ckpt)
        ckpt_state_dict = torch.load(args.ckpt)['model_state_dict']
        model.load_state_dict(ckpt_state_dict)
        model.cuda()
    test_sentences = [
        'people continue to enquire the reason for the race for outer space',
        'the secretariat is expected to race tomorrow']
        for i, sentence in enumerate(test_sentences):
            print('## Test case %d' % i)
            print('\tTest input:%s' % sentence)
            output = evaluate(model, sentence, train_dataset)
            print('\tTest output:', output)
