for split in 0;
  do for shots in 5;
    do for task in matching;
      do for engine in 'gpt-4';
        do python gpt4_from_description_answer.py --engine $engine --task $task --val 0 --split $split --shots $shots;
      done;
    done;
  done;
done