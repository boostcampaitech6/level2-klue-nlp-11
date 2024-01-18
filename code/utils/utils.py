import pandas as pd

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
    subject_start_dat_marker="<S:DAT>",
    subject_start_poh_marker="<S:POH>",
    subject_start_noh_marker="<S:NOH>",
    
    subject_end_per_marker ="</S:PER>",
    subject_end_org_marker ="</S:ORG>",
    subject_end_loc_marker ="</S:LOC>",
    subject_end_dat_marker ="</S:DAT>",
    subject_end_poh_marker ="</S:POH>",
    subject_end_noh_marker ="</S:NOH>",

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
TYPE_MARKERS_PUNC = dict(
    subject_start_per_marker="# * PER *",
    subject_start_org_marker="# * ORG *",
    subject_start_loc_marker="# * LOC *",
    subject_start_dat_marker="# * DAT *",
    subject_start_poh_marker="# * POH *",
    subject_start_noh_marker="# * NOH *",
    
    # subject_end_per_marker ="</S:PER>",
    # subject_end_org_marker ="</S:ORG>",
    # subject_end_loc_marker ="</S:LOC>",
    # subject_end_dat_marker ="</S:DAT>",
    # subject_end_poh_marker ="</S:POH>",
    # subject_end_noh_marker ="</S:NOH>",

    object_start_per_marker="@ * PER *",
    object_start_org_marker="@ * ORG *",
    object_start_loc_marker="@ * LOC *",
    object_start_dat_marker="@ * DAT *",
    object_start_poh_marker="@ * POH *",
    object_start_noh_marker="@ * NOH *",
    
    # object_end_per_marker ="</O:PER>",
    # object_end_org_marker ="</O:ORG>",
    # object_end_loc_marker ="</O:LOC>",
    # object_end_dat_marker ="</O:DAT>",
    # object_end_poh_marker ="</O:POH>",
    # object_end_noh_marker ="</O:NOH>",
)

def entity_marker(data : pd.Series):
    sent = data['sentence']
    sbj = data['subject_word']
    obj = data['object_word']
    sent = sent.replace(sbj, MARKERS['subject_start_marker']+' '+sbj+' '+MARKERS['subject_end_marker'])
    sent = sent.replace(obj, MARKERS['object_start_marker']+' '+obj+' '+MARKERS['object_end_marker'])
    return sent
# >>>'〈Something〉는 <OBJ> 조지 해리슨 </OBJ>이 쓰고 <SUB> 비틀즈 </SUB>가 1969년 앨범 《Abbey Road》에 담은 노래다.'
def typed_entity_marker(data : pd.Series):
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
# >>>'〈Something〉는 <O:PER> 조지 해리슨 </O:PER>이 쓰고 <S:ORG> 비틀즈 </S:ORG>가 1969년 앨범 《Abbey Road》에 담은 노래다.'
def typed_entity_marker_punc(data : pd.Series):
    sent = data['sentence']
    sbj = data['subject_word']
    sbj_type = data['subject_type']
    obj = data['object_word']
    obj_type = data['object_type']
		# Subject와 Object에 붙는 문장부호는 다르다!
    sent = sent.replace(sbj, f' @ * {sbj_type} * ' + sbj + ' @ ')
    sent = sent.replace(obj, f' # * {obj_type} * ' + obj + ' # ')
    return sent
# '〈Something〉는 # * PER * 조지 해리슨 #이 쓰고 @ * ORG * 비틀즈 @가 1969년 앨범 《Abbey Road》에 담은 노래다.'