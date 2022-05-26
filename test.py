import os
import deeplabcut
import torch

def deeplab():
    main_dir = os.getcwd()

    video_path = main_dir + "\cat2.mp4"
    print(video_path)

    project_name = 'testing_cats'
    your_name = 'william'
    model2use = 'full_cat'
    videotype = 'mp4'

    # Down sampling video to decrease input size 
    # video_path = deeplabcut.DownSampleVideo(video_path, width=300)

    config_path, train_config_path = deeplabcut.create_pretrained_project(
        project_name,
        your_name,
        [video_path],
        videotype=videotype,
        model=model2use,
        analyzevideo=True,
        createlabeledvideo=False,
        copy_videos=True, #must leave copy_videos=True
    )

    project_path = os.path.dirname(config_path)
    full_video_path = os.path.join(
        project_path,
        'videos',
        os.path.basename(video_path),
    )

    bodyparts = ['Nose', 'TailSet', 'L_F_Paw', 'R_F_Paw', 'L_B_Paw', 'R_B_Paw']
    deeplabcut.create_labeled_video(config_path, [full_video_path], videotype=videotype, save_frames=True, draw_skeleton=True, displayedbodyparts=bodyparts)
    deeplabcut.plot_trajectories(config_path, [full_video_path], displayedbodyparts=bodyparts)

def run():
    torch.multiprocessing.freeze_support()
    deeplab()

if __name__ == '__main__':
    run()