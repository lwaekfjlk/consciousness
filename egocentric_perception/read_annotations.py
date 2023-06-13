import json
import subprocess

def extract_frame(input_file, frame_number, output_file):
    command = [
        'ffmpeg',
        '-probesize', '50M',
        '-analyzeduration', '100M',
        '-i', input_file,
        '-vf', f'select=eq(n\,{frame_number})',
        '-vsync', 'vfr',
        output_file
    ]

    try:
        subprocess.run(command, check=True)
        print(f'Successfully extracted frame {frame_number} from {input_file} to {output_file}')
    except subprocess.CalledProcessError as e:
        print(f'An error occurred while extracting frame: {str(e)}')


if __name__ == '__main__':
    with open('./ego4d_data/v2/annotations/nlq_train.json') as f:
        data = json.load(f)

    video_annotations = data['videos']
    for annotation in video_annotations:
        video_uid = annotation['video_uid']
        if video_uid == '750c5383-58f6-4063-8cc8-50691a68994d':
            clips = annotation['clips']
            for clip in clips:
                clip_annotations = clip['annotations']
                for annotation in clip_annotations:
                    language_queries = annotation['language_queries']
                    for query in language_queries:
                        question = query['query']
                        answer = query['answer']
                        video_start_frame = query['video_start_frame']
                        video_end_frame = query['video_end_frame']
                        print(question, answer)
                        frame_num = (video_start_frame + video_end_frame) // 2
                        print(frame_num)
                        # Example usage
                        #extract_frame('750c5383-58f6-4063-8cc8-50691a68994d.mp4', frame_num, 'frame_{}.png'.format(frame_num))