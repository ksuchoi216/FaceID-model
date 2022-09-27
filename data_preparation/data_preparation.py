import cv2


class DataPreparation(object):
    def __init__(self, cfg):
        IsSavePhoto = cfg['IsSavePhoto']
        IsSaveFace = cfg['IsSaveFace']
        source_path = cfg['source_path']
        folder_name_for_video = cfg['folder_name_for_video']
        folder_to_save_photos = cfg['folder_to_save_photos']
        user_names = cfg['user_names']
        video_names = cfg['video_names']
        video_extension = cfg['video_extension']
        frame_count_limit = cfg['frame_count_limit']
        frame_interval_for_sampling = cfg['frame_interval_for_sampling']

        if IsSavePhoto and not os.path.exists(folder_to_save_photos):
            os.makedirs(folder_to_save_photos)

        for user_name, video_name in zip(user_names, video_names):
            path_for_video = folder_for_saved_videos
            path_for_video += video_name + video_extension
            path_for_username_folder = folder_to_save_photos + user_name+'/'
            print(f'{path_for_video} with {path_for_username_folder}')
        
        if not os.path.exists(path_for_username_folder):
            os.makedirs(path_for_username_folder)
            
        vc = cv2.VideoCapture(path_for_video)
        success, frame = vc.read()
        
        if not success:
            print("capturing frames failed")

        count = 0
        frame_count = 0

        while success:
            success, frame = vc.read()
            if frame is not None:
                h, w, ch = frame.shape

            frame = frame[int(h*(3/8)):int(h*(8/8)), int(w*(2/8)): int(w*(6/8))]

            if frame_count == frame_count_limit:
                break

            count += 1
            if count%frame_interval_for_sampling == 0:
                path_for_save = path_for_username_folder + str(frame_count) + '.jpg'
                cv2.imwrite(path_for_save, frame)
                success, frame = vc.read()
                print(f'[{count:5}][{frame_count:4}] A image was saved at {path_for_save}')
                frame_count += 1