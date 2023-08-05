from src.logger import logger
from src.exceptions import CustomException
import sys
from src.utils import load_pkl , dump_pkl
from src.extract_features import (extract_features , 
                                   model)

# from src.recommend import feature_list , file_names

img_path = load_pkl(file_path='model/img_files_path.pkl')

# images_features_from_model = feature_extractor(img_path=img_path,model=model)

img_features = extract_features(imgs_file_paths=img_path,model=model)

dump_pkl(obj=img_features,file_path='model/img_features.pkl')




# print(feature_list)
# print(len(file_names))






































# if __name__=="__main__":
#     try:
#         a = 1/0
#         # raise CustomException
#     except Exception as e:
#         logger.info('ZERO Division ERRor')
#         raise CustomException(e,sys)