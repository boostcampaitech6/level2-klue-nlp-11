import pandas as pd
# train 및 test 데이터 불러오기
train_data = pd.read_csv("./train/train.csv")
test_data = pd.read_csv("./test/test_data.csv")

MARKERS = dict(
    subject_start_marker="<SUB>",
    subject_end_marker  ="</SUB>",
    object_start_marker ="<OBJ>",
    object_end_marker   ="</OBJ>",
)
TYPE_MARKERS = dict(
    subject_start_per_marker="<S:PER>",
    subject_start_org_marker="<S:ORG>",
    subject_start_loc_marker="<S:LOC>",
    subject_end_per_marker ="</S:PER>",
    subject_end_org_marker ="</S:ORG>",
    subject_end_loc_marker="</S:LOC>",
    object_start_per_marker="<O:PER>",
    object_start_org_marker="<O:ORG>",
    object_start_loc_marker="<O:LOC>",
    object_start_dat_marker="<O:DAT>",
    object_start_poh_marker="<O:POH>",
    object_start_noh_marker="<O:NOH>",
    object_end_per_marker ="</O:PER>",
    object_end_org_marker ="</O:ORG>",
    object_end_loc_marker ="</O:LOC>",
    object_end_dat_marker ="</O:DAT>",
    object_end_poh_marker ="</O:POH>",
    object_end_noh_marker ="</O:NOH>",
)

# Entity 정보 분리 - train
# 분리된 entity를 저장할 리스트
subject_word = []
subject_start = []
subject_end = []
subject_type = []

object_word = []
object_start = []
object_end = []
object_type = []

# train data에 데이터 추가
from ast import literal_eval
for idx, row in train_data.iterrows():
    # print(idx)
    sub_data = literal_eval(row['subject_entity']) # type==dict
    obj_data = literal_eval(row['object_entity'])
    
    subject_word.append(sub_data['word'])
    subject_start.append(int(sub_data['start_idx']))
    subject_end.append(int(sub_data['end_idx']))
    subject_type.append(sub_data['type'])
    
    object_word.append(obj_data['word'])
    object_start.append(int(obj_data['start_idx']))
    object_end.append(int(obj_data['end_idx']))
    object_type.append(obj_data['type'])

# 데이터 추가
train_data['subject_word'] = subject_word
train_data['subject_start'] = subject_start
train_data['subject_end'] = subject_end
train_data['subject_type'] = subject_type

train_data['object_word'] = object_word
train_data['object_start'] = object_start
train_data['object_end'] = object_end
train_data['object_type'] = object_type

# 미사용 컬럼 삭제 -> load_data.py에서 entity를 사용함
#train_data = train_data.drop(['subject_entity', 'object_entity'], axis=1)

# Entity 정보 분리 - test
# 분리된 entity를 저장할 리스트
subject_word = []
subject_start = []
subject_end = []
subject_type = []

object_word = []
object_start = []
object_end = []
object_type = []

# train data에 데이터 추가
from ast import literal_eval
for idx, row in test_data.iterrows():
    # print(idx)
    sub_data = literal_eval(row['subject_entity']) # type==dict
    obj_data = literal_eval(row['object_entity'])
    
    subject_word.append(sub_data['word'])
    subject_start.append(int(sub_data['start_idx']))
    subject_end.append(int(sub_data['end_idx']))
    subject_type.append(sub_data['type'])
    
    object_word.append(obj_data['word'])
    object_start.append(int(obj_data['start_idx']))
    object_end.append(int(obj_data['end_idx']))
    object_type.append(obj_data['type'])

# 데이터 추가
test_data['subject_word'] = subject_word
test_data['subject_start'] = subject_start
test_data['subject_end'] = subject_end
test_data['subject_type'] = subject_type

test_data['object_word'] = object_word
test_data['object_start'] = object_start
test_data['object_end'] = object_end
test_data['object_type'] = object_type

# 미사용 컬럼 삭제 -> load_data.py에서 entity를 사용함
#test_data = test_data.drop(['subject_entity', 'object_entity'], axis=1)

# 마커 추가 함수 설정
def entity_marker(data : pd.Series):
    # 예시: 〈Something〉는 <OBJ> 조지 해리슨 </OBJ>이 쓰고 <SUB> 비틀즈 </SUB>가 1969년 앨범 《Abbey Road》에 담은 노래다.
    sent = data['sentence']
    sbj = data['subject_word']
    obj = data['object_word']
    sent = sent.replace(sbj, MARKERS['subject_start_marker']+' '+sbj+' '+MARKERS['subject_end_marker'])
    sent = sent.replace(obj, MARKERS['object_start_marker']+' '+obj+' '+MARKERS['object_end_marker'])
    return sent

def typed_entity_marker(data : pd.Series):
    # 예시: 〈Something〉는 <O:PER> 조지 해리슨 </O:PER>이 쓰고 <S:ORG> 비틀즈 </S:ORG>가 1969년 앨범 《Abbey Road》에 담은 노래다.
    sent = data['sentence']
    sbj = data['subject_word']
    sbj_start_type_mark = TYPE_MARKERS[f"subject_start_{data['subject_type'].lower()}_marker"]
    sbj_end_type_mark = TYPE_MARKERS[f"subject_end_{data['subject_type'].lower()}_marker"]
    obj = data['object_word']
    obj_start_type_mark = TYPE_MARKERS[f"object_start_{data['object_type'].lower()}_marker"]
    obj_end_type_mark = TYPE_MARKERS[f"object_end_{data['object_type'].lower()}_marker"]
    sent = sent.replace(sbj, sbj_start_type_mark+' '+sbj+' '+sbj_end_type_mark)
    sent = sent.replace(obj, obj_start_type_mark+' '+obj+' '+obj_end_type_mark)
    return sent

def typed_entity_marker_punc(data : pd.Series):
    # 예시: 〈Something〉는 # * PER * 조지 해리슨 # 이 쓰고 @ * ORG * 비틀즈 @ 가 1969년 앨범 《Abbey Road》에 담은 노래다.
    sent = data['sentence']
    sbj = data['subject_word']
    sbj_type = data['subject_type']
    obj = data['object_word']
    obj_type = data['object_type']
    sent = sent.replace(sbj, '@'+f' * {sbj_type} * '+sbj+' @ ')
    sent = sent.replace(obj, '#'+f' * {obj_type} * '+obj+' # ')
    return sent

# train 데이터 전처리 후 저장
entity_marker_list = []
typed_entity_marker_list = []
typed_entity_marker_punc_list = []
for i in range(len(train_data)):
    entity_marker_list.append(entity_marker(train_data.iloc[i]))
    typed_entity_marker_list.append(typed_entity_marker(train_data.iloc[i]))
    typed_entity_marker_punc_list.append(typed_entity_marker_punc(train_data.iloc[i]))
    
train_data['sentence'] = entity_marker_list
train_data.to_csv("./train/train_entity_marker.csv")
train_data['sentence'] = typed_entity_marker_list
train_data.to_csv("./train/train_typed_entity_marker.csv")
train_data['sentence'] = typed_entity_marker_punc_list
train_data.to_csv("./train/train_typed_entity_marker_punc.csv")

# test 데이터 전처리 후 저장
entity_marker_list = []
typed_entity_marker_list = []
typed_entity_marker_punc_list = []
for i in range(len(test_data)):
    entity_marker_list.append(entity_marker(test_data.iloc[i]))
    typed_entity_marker_list.append(typed_entity_marker(test_data.iloc[i]))
    typed_entity_marker_punc_list.append(typed_entity_marker_punc(test_data.iloc[i]))
    
test_data['sentence'] = entity_marker_list
test_data.to_csv("./test/test_entity_marker.csv")
test_data['sentence'] = typed_entity_marker_list
test_data.to_csv("./test/test_typed_entity_marker.csv")
test_data['sentence'] = typed_entity_marker_punc_list
test_data.to_csv("./test/test_typed_entity_marker_punc.csv")