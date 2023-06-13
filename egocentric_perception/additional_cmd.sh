ffmpeg -ss 00:00:38 -i demo.mp4 -t 5 -vn -acodec copy frame3.wav
ffmpeg -ss 00:01:10 -i demo.mp4 -t 5 -vn -acodec copy frame2.wav
ffmpeg -ss 00:01:46 -i demo.mp4 -t 5 -vn -acodec copy frame1.wav