for split in {0,1,2,3,4};
  do for shots in 5;
    do for task in matching;
      do for engine in 'gpt-3.5-turbo';
        do python gpt4_from_description.py --engine $engine --task $task --val 0 --split $split --shots $shots;
      done;
    done;
  done;
done