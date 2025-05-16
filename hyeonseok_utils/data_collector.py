from hyeonseok_utils.gif_list_load import gif_list_load
from hyeonseok_utils.check_data_already_exists_recursive import check_data_path_already_exists_recursive
import os
from pathlib import Path

# 현재 빠진것, 현재 스텝에 대해서 Result가 이미 있는 것을 확인 하는 로직을 추가하지는 않음

def process_data_list_loader(SELECTOR_FILE_PATH, MAX_SAMPLES, DATA_DIR, AVAILABLE_DATA_FORMATS, SPLITOR, ENCODING):
    # 현재는 로컬에서 path를 확인함으로 바로 gif_list_load을 사용하지 않음
    # data_files = gif_list_load(DATA_DIR)
    # 추후 데이터베이스를 도입하면 사용할수도 있음.
    raw_data_path, missing_path_list = find_data_name_from_folder(
        data_name_list = data_name_list_parser_from_file(
            data_select_list_file_path = SELECTOR_FILE_PATH,
            MAX_SAMPLES=MAX_SAMPLES,
            ENCODING=ENCODING,
            splitor=SPLITOR
        ),
        data_dir = DATA_DIR,
        request_file_format_list = AVAILABLE_DATA_FORMATS,
    )
    
    print("Missing files:", missing_path_list)
    print("Please check the files in the data directory")
    
    data_path_list, missing_path_list2 = check_data_path_already_exists_recursive(
        raw_data_path,
        DATA_DIR,
        recursive=True
    )
    return data_path_list, missing_path_list

# 아래의 함수는 어떤 데이터를 선별할지 기록된 파일을 파씽하는 함수입니다.
def data_name_list_parser_from_file(
    data_select_list_file_path,
    MAX_SAMPLES=-1,
    ENCODING='utf-8',
    splitor=','
    # request_file_format_list = [".gif"]
    ):
    with open(data_select_list_file_path, "r", encoding=ENCODING) as f:
        selected_file = f.read().split(splitor)
        # print(f.read())
        selected_file = [file.strip() for file in selected_file]
    selected_file = selected_file[0: MAX_SAMPLES]
    print(f"Successfully selected {len(selected_file)} files for processing")
    return selected_file


# 확장자 없이 파일의 이름만 받음
# 기존에 존재하는 인풋 데이터를 리스트 형태로 받은후 이곳에서 검색을 진행합니다.
# 만약에 존재하지 않는다면 missing data로 처리합니다.
# 만약 존재한다면 result에 추가하여 반환합니다.
def find_data_name_from_folder(
    data_name_list,
    data_dir,
    request_file_format_list = [".gif", ".webm"],
    ):
    available_file_format_list= [".gif", ".webm"]
    for ext in request_file_format_list:
        if ext not in available_file_format_list:
            raise ValueError(
                f"❌ 허용되지 않은 파일 형식입니다: {ext} (지원 확장자: {available_file_format_list})"
            )
            
    missing_files = []
    data_path_list = []
    for file_name in data_name_list:
        flag = False
        for ext in request_file_format_list:
            data_path_candidate = os.path.join(data_dir, file_name.strip()) + ext 
            if os.path.exists(data_path_candidate):
                flag = True
                data_path_list.append(str(data_path_candidate))
                break
        if not flag:
            missing_files.append(file_name.strip())
            print(f"❌ 요청한 파일이 존재하지 않습니다: {os.path.join(data_dir, file_name.strip())+ext}")
    
    return data_path_list, missing_files