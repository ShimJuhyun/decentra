# -*- coding: utf-8 -*-
"""
Local Scorecard Explainer Prototype.

- 2025.05.29 / 심주현
- 여기서의 Scorecard 는 "평점표 형태로 설명하겠다" 라는 의미.
- 평점표 모형과는 무관.(방법론만 차용)
- 1차 배포: 2025.05.30

"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Union, Any, Optional, Literal
import numpy as np
import pandas as pd

from itertools import combinations


def _get_lift_result(data_row, comb_no, scorecard_explainer, lift_factor_index, scorecard_dict):
    lift_factor = scorecard_explainer.loc[lift_factor_index]
    feature_name = lift_factor['feature']
    state = scorecard_dict[feature_name]

    original_data_value = data_row[feature_name]
    original_recode = state.recode

    new_recode = lift_factor['recode']

    lift_factor_recode_index = state.recodes.index(lift_factor['recode'])
    _min, _max = state.ranges[lift_factor_recode_index]
    
    # new_data_value = state.representative_values[lift_factor_recode_index]
    new_data_value = _min if abs(original_data_value - _min) < abs(original_data_value - _max) else _max
    new_data_row = data_row.copy()
    new_data_row[feature_name] = new_data_value

    result_row = {'no': comb_no,
                'feature': feature_name,
                'description': state.description,
                'From Recode': original_recode,
                'New Recode': new_recode,
                'From Value': original_data_value,
                'To Value': new_data_value,
                }
    
    return new_data_row, result_row


# [xxx] 설명 우선순위 등등 추가 필요.
@dataclass
class PriorFetureInfo:
    name: str = field()
    description: str = field(default='')
    ranges: List[tuple] = field(default_factory=list())
    recodes: List[int] = field(default_factory=list())
    representative_values: Optional[Union[int, float, str]] = field(default=None)

    def __post_init__(self):
        # self._recode_to_weight = dict(zip(self.recodes, self.weights))
        if self.representative_values is None:
            self.representative_values = [(rng[0] + rng[1])/2  for rng in self.ranges]

        self.recode_to_range = dict(zip(self.recodes, self.ranges))

    def get_recode(self, value):
        for i, rng in enumerate(self.ranges):
            if rng[0] <= value <= rng[1]:
                return self.recodes[i]
            
    def to_dict(self):
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str,  Any]):
        return cls(**data)
    

@dataclass    
class LocalExplainerState(PriorFetureInfo):
    scores: List[Union[int, float]] = field(default_factory=list)
    score_mean: float = field(default=0)
    relative_scores: List[float] = field(default_factory=list)
    scaled_scores: List[float] = field(default_factory=list)

    recode:Optional[int] = field(default=None)

    @classmethod
    def create(cls, feature_info:PriorFetureInfo):
        return cls(**feature_info.to_dict())
    
    def update(self):
        if len(self.scores) > 0:
            self.score_mean = float(np.mean(self.scores))
            self.relative_scores = [round(float(value), 2) for value in (np.array(self.scores) - self.score_mean)]
            base_score = self.scores[self.recodes.index(self.recode)]
            self.scaled_scores = [round(float(value),2) for value in (np.array(self.scores) - base_score)]


def _get_weight(value, thresholds, weights):
    for i, tresh in enumerate(thresholds):
        if value <= tresh:
            return weights[i]
    return weights[-1]

class TempModel:
    def predict(self, data_row):
        score = 0
        score += _get_weight(data_row['feature001'], [0], [0,100])
        score += _get_weight(data_row['feature002'], [0, 1, 2], [0, 30, 50, 90])
        score += _get_weight(data_row['feature003'], [0, 2], [0, 70, 120])

        return score
    
    
# %% 



@dataclass
class LocalScorecardExplainer:
    model: Any
    scorecard: List[PriorFetureInfo]
    states: List[LocalExplainerState] = field(default_factory=list)
    
    def explain(self, data_row:Dict[str, Union[int, float, str]], _type:Literal['rc', 'lift', 'all']='all'):
        self.states = []
        org_score = self.model.predict(data_row)
        for feature_info in self.scorecard:
            state = LocalExplainerState.create(feature_info)
            feature_name = state.name

            for j, recode in enumerate(state.recodes):
                if state.get_recode(data_row[feature_name]) == recode:
                    state.recode = recode
                    state.scores.append(org_score)
                else:
                    changed_data_row = data_row.copy()
                    changed_data_row[feature_name] = state.representative_values[j]
                    state.scores.append(self.model.predict(changed_data_row))

            state.update()
            self.states.append(state)

        # 사유코드 생성
        unstack_index = []
        bucket = []
        for state in self.states:
            for j, recode in enumerate(state.recodes):
                row = {'feature': state.name,
                       'description': state.description,
                       'recode': state.recodes[j],
                       'range': '{0}~{1}'.format(state.ranges[j][0], state.ranges[j][1]),
                       'score': state.scores[j],
                       'is_real': 'T' if state.recode == recode else 'F',
                       'relative': state.relative_scores[j],
                       'scaled': state.scaled_scores[j],
                       }
                bucket.append(row)
                unstack_index.append((state.name, recode))

        scorecard_dict = {feature_info.name: feature_info for feature_info in self.states}

        scorecard_explainer = pd.DataFrame(bucket)
        scorecard_explainer.insert(0, 'no', scorecard_explainer.index + 1)

        pos, neg = scorecard_explainer.query('relative > 0').sort_values('relative', ascending=False), scorecard_explainer.query('relative <= 0').sort_values('relative')
        pos['rc'] = pos.index.map(lambda x: 'P{0}'.format(str(x+1).zfill(3)))
        neg['rc'] = neg.index.map(lambda x: 'N{0}'.format(str(x+1).zfill(3)))
        scorecard_explainer = pd.concat([pos, neg]).sort_values('no').reset_index(drop=True)

        self.explaination = scorecard_explainer

        if _type == 'rc':
            return scorecard_explainer

        lift_factors =  scorecard_explainer.query('scaled > 0').index.to_list()
        lift_factors_combination = list(combinations(lift_factors, 1)) + list(combinations(lift_factors, 2)) + list(combinations(lift_factors, 3))

        lift_factor_bucket = []
        for comb_no, lift_factor_index in enumerate(lift_factors_combination):
            new_data_row = data_row.copy()
            result_row = []
            for j, _idx in enumerate(lift_factor_index):
                new_data_row, _result_row = _get_lift_result(new_data_row, comb_no, scorecard_explainer, _idx, scorecard_dict)
                result_row.append(_result_row)

            # 피처 중복
            if len(result_row) >= 2:
                if result_row[0]['feature'] == result_row[1]['feature']:
                    continue

            if len(result_row) == 3:
                if result_row[1]['feature'] == result_row[2]['feature']:
                    continue

            for _result_row in result_row:
                _result_row['From Score'] = org_score
                _result_row['To Score'] = self.model.predict(new_data_row)
                
                if hasattr(self.model, 'get_grade'):
                    _result_row['From Grade'] = self.model.get_grade(_result_row['From Score'])
                    _result_row['To Grade'] = self.model.get_grade(_result_row['To Score'])
                
                lift_factor_bucket.append(_result_row)

        lift_factor = pd.DataFrame(lift_factor_bucket).sort_values(['To Score', 'no'], ascending=[False, True])

        if _type == 'lift':
            return lift_factor
        else:
            return scorecard_explainer, lift_factor
            
def local_scorecard_explain(model, prior_json, data_row, _type):
    prior_feature_info_json = [PriorFetureInfo(**info) for info in prior_json]
    explainer = LocalScorecardExplainer(model, prior_feature_info_json)
    return explainer.explain(data_row, _type)
    
    