import os.path as osp
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence
import pandas.api.types


import numpy as np
import pandas as pd
import torch

from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger, print_log


from mm_custom.registry import METRICS

from prettytable import PrettyTable

from tqdm import tqdm
import os

import sklearn.metrics


# MetricV2
@METRICS.register_module()
class Metric(BaseMetric):
    
    def __init__(self,
                 metainfo,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,                 
                 **kwargs) -> None:


        self.metainfo = metainfo
        self.train_weight_df = pd.read_csv(f'./data/processed/train_weight.csv')

        self.binary_class_names = ['bowel', 'extravasation']
        self.triple_class_names = ['kidney', 'liver', 'spleen']
        self.any_injury_class_names = ['any_injury']

        logger: MMLogger = MMLogger.get_current_instance()
        self.work_dir =  os.path.dirname(logger.log_file)

        self.binary_injury_types = ['healthy', 'injury']
        self.triple_injury_types = ['healthy', 'low', 'high']

        self.all_class_names = self.binary_class_names + self.triple_class_names + self.any_injury_class_names
        self.metric_class_names = self.metainfo['metric_class_names']
        self.class_names = self.metainfo['class_names']

        super().__init__(collect_device=collect_device, prefix=prefix)




    def process(self, data_batch, data_samples: dict) -> None:


        patient_ids = data_batch['data_batch']['patient_id']
        series_ids = data_batch['data_batch']['series_id']


        for  patient_id, series_id, data_sample in zip(patient_ids, series_ids, data_samples):

            for k, v in data_sample.items():
                for name, value in v.items():
                    if isinstance(value, torch.Tensor):
                        data_sample[k][name] = value.cpu().detach().numpy()

            result = {'patient_id': patient_id, 'series_id': series_id}
            result.update(data_sample)

            self.results.append(result)

        

    def compute_metrics(self, results: list) -> Dict[str, float]:
        
        logger: MMLogger = MMLogger.get_current_instance()


        patient_ids, series_ids = [], []
        pred_probs = []
        class_names = self.class_names

        for result in results:
            patient_id, series_id = result['patient_id'], result['series_id']

            pred_prob_dic = result['pred_prob_dic']

            pred_prob = []
            for name in class_names:
                pred_prob_organ = pred_prob_dic[name]
                pred_prob.append(pred_prob_organ)
            pred_prob = np.concatenate(pred_prob, axis=0)

            pred_probs.append(pred_prob)
            patient_ids.append(patient_id)
            series_ids.append(series_id)


        pred_probs = np.array(pred_probs)
        patient_ids = np.array(patient_ids).reshape(-1, 1)
        series_ids = np.array(series_ids).reshape(-1, 1)

        data = np.concatenate([patient_ids, series_ids, pred_probs], axis=1) # [N, col_nums]

        column_names = ['patient_id', 'series_id']

        for name in class_names:
            if name in self.binary_class_names:
                injury_types = self.binary_injury_types
            elif name in self.triple_class_names:
                injury_types = self.triple_injury_types
            else:
                raise NotImplementedError
            
            for injury_type in injury_types:
                column_names.append(f'{name}_{injury_type}')

        pred_df = pd.DataFrame(data=data, columns=column_names)
        pred_df['patient_id'] = pred_df['patient_id'].astype(int)
        pred_df['series_id'] = pred_df['series_id'].astype(int)


        pred_df = pred_df.sort_values(['patient_id', 'series_id']).reset_index(drop=True)

        pred_df.to_csv(os.path.join(self.work_dir, 'pred.csv'), index=None)


        del pred_df['series_id']
        pred_df = pred_df.groupby(['patient_id'], as_index=False).mean()
        gt_df = self.train_weight_df[self.train_weight_df.patient_id.isin(pred_df.patient_id.values)].reset_index(drop=True).copy()

        gt_df = gt_df.sort_values(['patient_id']).reset_index(drop=True)

        only_binary = False
        only_triple = False

        if len(class_names) == 2:
            only_binary = True
        if len(class_names) == 3:
            only_triple = True

        losses = eval_score(gt_df, pred_df, row_id_column_name='patient_id', only_binary=only_binary, only_triple=only_triple)

        


        log_loss_dict = {}

        for i, name in enumerate(class_names):
            log_loss_dict[name] = losses[i]
        
        if not only_binary and not only_triple:
            log_loss_dict['any_injury'] = losses[-1]

        log_loss_dict['mean'] = np.mean(np.array(list(log_loss_dict.values())))

        

        metrics = log_loss_dict


        class_table_data = PrettyTable()

        for key, val in metrics.items():
            #class_table_data.add_row([key, [val]])
            class_table_data.add_column(key, [val])

        print_log('per class results:', logger)
        print_log('\n' + class_table_data.get_string(), logger=logger)

        return metrics




def eval_score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str, only_binary, only_triple):
    '''
    Pseudocode:
    1. For every label group (liver, bowel, etc):
        - Normalize the sum of each row's probabilities to 100%.
        - Calculate the sample weighted log loss.
    2. Derive a new any_injury label by taking the max of 1 - p(healthy) for each label group
    3. Calculate the sample weighted log loss for the new label group
    4. Return the average of all of the label group log losses as the final score.
    '''
    del solution[row_id_column_name]
    del submission[row_id_column_name]

    # Run basic QC checks on the inputs
    if not pandas.api.types.is_numeric_dtype(submission.values):
        raise ParticipantVisibleError('All submission values must be numeric')

    if not np.isfinite(submission.values).all():
        raise ParticipantVisibleError('All submission values must be finite')

    if solution.min().min() < 0:
        raise ParticipantVisibleError('All labels must be at least zero')
    if submission.min().min() < 0:
        raise ParticipantVisibleError('All predictions must be at least zero')

    # Calculate the label group log losses
    binary_targets = ['bowel', 'extravasation']
    triple_level_targets = ['kidney', 'liver', 'spleen']
    all_target_categories = binary_targets + triple_level_targets

    if only_binary and only_triple:
        raise NotImplementedError

    if only_binary:
        all_target_categories = binary_targets

    if only_triple :
        all_target_categories = triple_level_targets

    label_group_losses = []
    for category in all_target_categories:
        if category in binary_targets:
            col_group = [f'{category}_healthy', f'{category}_injury']
        else:
            col_group = [f'{category}_healthy', f'{category}_low', f'{category}_high']

        solution = normalize_probabilities_to_one(solution, col_group)

        for col in col_group:
            if col not in submission.columns:
                raise ParticipantVisibleError(f'Missing submission column {col}')
        submission = normalize_probabilities_to_one(submission, col_group)
        label_group_losses.append(
            sklearn.metrics.log_loss(
                y_true=solution[col_group].values,
                y_pred=submission[col_group].values,
                sample_weight=solution[f'{category}_weight'].values
            )
        )


    if not only_binary and not only_triple:

        # Derive a new any_injury label by taking the max of 1 - p(healthy) for each label group
        healthy_cols = [x + '_healthy' for x in all_target_categories]
        any_injury_labels = (1 - solution[healthy_cols]).max(axis=1)
        any_injury_predictions = (1 - submission[healthy_cols]).max(axis=1)
        any_injury_loss = sklearn.metrics.log_loss(
            y_true=any_injury_labels.values,
            y_pred=any_injury_predictions.values,
            sample_weight=solution['any_injury_weight'].values
        )

        label_group_losses.append(any_injury_loss)

    #mean_loss = np.mean(label_group_losses)
    #losses = label_group_losses + [mean_loss]

    return label_group_losses





class ParticipantVisibleError(Exception):
    pass


def normalize_probabilities_to_one(df: pd.DataFrame, group_columns: list) -> pd.DataFrame:
    # Normalize the sum of each row's probabilities to 100%.
    # 0.75, 0.75 => 0.5, 0.5
    # 0.1, 0.1 => 0.5, 0.5
    row_totals = df[group_columns].sum(axis=1)
    if row_totals.min() == 0:
        raise ParticipantVisibleError('All rows must contain at least one non-zero prediction')
    for col in group_columns:
        df[col] /= row_totals
    return df


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    '''
    Pseudocode:
    1. For every label group (liver, bowel, etc):
        - Normalize the sum of each row's probabilities to 100%.
        - Calculate the sample weighted log loss.
    2. Derive a new any_injury label by taking the max of 1 - p(healthy) for each label group
    3. Calculate the sample weighted log loss for the new label group
    4. Return the average of all of the label group log losses as the final score.
    '''
    del solution[row_id_column_name]
    del submission[row_id_column_name]

    # Run basic QC checks on the inputs
    if not pandas.api.types.is_numeric_dtype(submission.values):
        raise ParticipantVisibleError('All submission values must be numeric')

    if not np.isfinite(submission.values).all():
        raise ParticipantVisibleError('All submission values must be finite')

    if solution.min().min() < 0:
        raise ParticipantVisibleError('All labels must be at least zero')
    if submission.min().min() < 0:
        raise ParticipantVisibleError('All predictions must be at least zero')

    # Calculate the label group log losses
    binary_targets = ['bowel', 'extravasation']
    triple_level_targets = ['kidney', 'liver', 'spleen']
    all_target_categories = binary_targets + triple_level_targets

    label_group_losses = []
    for category in all_target_categories:
        if category in binary_targets:
            col_group = [f'{category}_healthy', f'{category}_injury']
        else:
            col_group = [f'{category}_healthy', f'{category}_low', f'{category}_high']

        solution = normalize_probabilities_to_one(solution, col_group)

        for col in col_group:
            if col not in submission.columns:
                raise ParticipantVisibleError(f'Missing submission column {col}')
        submission = normalize_probabilities_to_one(submission, col_group)
        label_group_losses.append(
            sklearn.metrics.log_loss(
                y_true=solution[col_group].values,
                y_pred=submission[col_group].values,
                sample_weight=solution[f'{category}_weight'].values
            )
        )

    # Derive a new any_injury label by taking the max of 1 - p(healthy) for each label group
    healthy_cols = [x + '_healthy' for x in all_target_categories]
    any_injury_labels = (1 - solution[healthy_cols]).max(axis=1)
    any_injury_predictions = (1 - submission[healthy_cols]).max(axis=1)
    any_injury_loss = sklearn.metrics.log_loss(
        y_true=any_injury_labels.values,
        y_pred=any_injury_predictions.values,
        sample_weight=solution['any_injury_weight'].values
    )

    label_group_losses.append(any_injury_loss)
    mean_loss = np.mean(label_group_losses)
    losses = label_group_losses + [mean_loss]

    return losses


