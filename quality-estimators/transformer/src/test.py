import torch
from tqdm import tqdm
import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import theilslopes, pearsonr

from . import utils
from .loader import *
from .model import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f'Device is {device}')

warnings.filterwarnings("ignore", category=FutureWarning)

def estimate_quality(dataloader, attn_matrices, cols=(['HB_1', 'HB_2'], ['time', 'seq_id', 'night'], ['majority'])):
    """
    Estimate the quality of the model's predictions and save the results to a CSV file.

    :param dataloader: Dataloader providing batches of input data and labels.
    :param attn_matrices: Attention matrices obtained from the model.
    :param cols: Tuple containing lists of feature columns, time-related columns, and target columns.
    """
    all_data = []
    X_cols, t_cols, y_cols = cols

    for (feats, labels), attn_matrix in zip(dataloader, attn_matrices):
        feats = feats.cpu().numpy()
        labels = labels.cpu().numpy()
        noise = attn_matrix.cpu().numpy() #(batch_size, seq_length, d_model)

        batch_size, seq_len, _ = feats.shape

        for batch_idx in range(batch_size):
            for seq_idx in range(seq_len):
                row_data = {}

                for i, col_name in enumerate(X_cols + t_cols):
                    row_data[col_name] = feats[batch_idx, seq_idx, i]

                for i, y_col in enumerate(y_cols):
                    row_data[y_col] = labels[batch_idx, seq_idx, i]

                for i, X_col in enumerate(X_cols):
                    row_data[f'noise_{X_col}'] = noise[batch_idx, seq_idx, i]

                all_data.append(row_data)

    df = pd.DataFrame(all_data)

    path = utils.get_path('..', '..', 'data', 'proc', filename=f'estim_transformer.csv')
    df.to_csv(path, index=False)

def test(data, classes, criterion, model, visualize=False, estimate=False):
    """
    Test the model on the provided data and calculate the test loss.

    :param data: Data to test the model on.
    :param criterion: Loss function used to compute the test loss.
    :param model: The model to be evaluated.
    :param visualize: Whether to visualize the model's predictions.
    :param estimate: Whether to estimate the quality of the predictions.
    """
    mfn = utils.get_path('..', '..', 'models', filename='transformer.pth')

    model.load_state_dict(torch.load(mfn))
    model.to(device)
    model.eval()

    batches = len(data)
    total_test_loss = 0.0
    true_values, pred_values = [], []
    attn_matrices = []

    progress_bar = tqdm(enumerate(data), total=batches, desc=f'Evaluation', leave=True)

    with torch.no_grad():
        for _, (X, y) in progress_bar:
            X = X.to(device)

            X, t = separate(src=X, c=[0,1], t=[2])
            X = merge(c=X, t=t)

            y_pred, attn_matrix = model(X)

            attn_matrices.append(attn_matrix)

            batch_size, seq_len, num_classes = y_pred.size()
            y_pred = y_pred.reshape(batch_size * seq_len, num_classes)
            y = y.reshape(batch_size * seq_len)

            test_loss = criterion(y_pred, y)

            total_test_loss += test_loss.item()
            progress_bar.set_postfix(Loss=test_loss.item())

            true_values.append(y.cpu().numpy())
            pred_values.append(y_pred.detach().cpu().numpy())
        
        true_values = np.concatenate(true_values)
        pred_values = np.concatenate(pred_values)

        pred_classes = np.argmax(pred_values, axis=1)

        avg_test_loss = total_test_loss / batches

        if estimate:
            estimate_quality(data, attn_matrices)

        if visualize:
            utils.visualize(type='heatmap',
                        values=(true_values, pred_classes), 
                        labels=('True Values', 'Predicted Values'), 
                        title='Train Heatmap ',
                        classes=classes,
                        coloring=['azure', 'darkblue'],
                        path=utils.get_dir('static', 'transformer'))

    logger.info(f'\nTesting complete!\nTesting Loss: {avg_test_loss:.6f}\n')

def main():
    """
    Main function to execute the testing workflow, including data preparation and model evaluation.
    """
    samples, chunks = 7680, 32
    seq_len = samples // chunks

    bitbrain_dir = utils.get_dir('..', '..', 'data', 'bitbrain')
    raw_dir = utils.get_dir('..', '..', 'data', 'raw')

    get_boas_data(base_path=bitbrain_dir, output_path=raw_dir)
    
    datapaths = split_data(dir=raw_dir, train_size=43, val_size=3, test_size=10)
    
    train_df, _, test_df = get_dataframes(datapaths, seq_len=seq_len, exist=True)
    _, weights = extract_weights(train_df, label_col='majority')

    classes = list(weights.keys())
    logger.info(f'Weights: {weights}.')

    datasets = create_datasets(dataframes=(test_df,), seq_len=seq_len)

    dataloaders = create_dataloaders(datasets, batch_size=512, drop_last=True)

    model = Transformer(in_size=3,
                        out_size=len(classes),
                        num_heads=1,
                        dropout=0.5)
    
    test(data=dataloaders[0],
         classes=classes,
         criterion=utils.WeightedCrossEntropyLoss(weights),
         model=model,
         visualize=True,
         estimate=True)

if __name__ == '__main__':
    main()