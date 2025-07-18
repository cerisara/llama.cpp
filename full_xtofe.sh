# make GGML_CUDA=1 llama-eval-callback

modnom="/home/data/models--Qwen--Qwen2.5-72B-Instruct-GGUF/snapshots/qwen2.5-72b-instruct-q4_k_m-00001-of-00012.gguf"
modnom="/home/data/Qwen2.5-72B-Instruct-GGUF/qwen2.5-72b-instruct-q3_k_m-00001-of-00009.gguf"

echo 'l_out-10' > layers2save
echo 'l_out-11' >> layers2save
echo 'l_out-12' >> layers2save
echo 'l_out-27' >> layers2save

docpp01="false"
docpp02="true"
docpp03="true"
docpp04="true"
docpp05="true"
docpp06="true"
docpp07="true"
docpp08="true"
docpp09="true"
docpp10="true"

dotrain02="false"
dotrain03="true"
dotrain04="true"
dotrain05="true"
dotrain06="true"
dotrain07="true"
dotrain08="true"
dotrain09="true"
dotrain10="true"

dotest02="true"
dotest03="true"
dotest04="true"
dotest05="true"
dotest06="true"
dotest07="true"
dotest08="true"
dotest09="true"
dotest10="true"

#
### PART 01
#

echo "\n\n#\n###\n#\n################### PART 01 ###################\n#\n###\n#\n\n"

if $docpp01; then

	tf="/home/data/fresh_news_part_10.txt"

	rm -f activs.txt activs.bin
	touch activs.txt

	while IFS="" read -r p || [ -n "$p" ]
	do
	        rm -rf detlog
	        mkdir detlog
	        ./llama-cli --logdir detlog --temp 0 -c 2048 -nkvo -m "$modnom" -p "$p" -fa -ngl 75 -n 1
	        grep prompt_token detlog/* | cut -c17- | sed 's/,//g;s,],,g' >> activs.txt
	done < "$tf"

	mv activs.txt activs_part1.txt
	mv activs.bin activs_part1.bin

else
        echo "\nSkipped cpp part 01\n"
fi


#
### PART 02
#
#
echo "\n\n#\n###\n#\n################### PART 02 ###################\n#\n###\n#\n\n"

if $docpp02; then 

	tf="/home/data/fresh_news_part_01.txt"

	rm -f activs.txt activs.bin
	touch activs.txt

	while IFS="" read -r p || [ -n "$p" ]
	do
		rm -rf detlog
		mkdir detlog
		./llama-cli --logdir detlog --temp 0 -c 2048 -nkvo -m "$modnom" -p "$p" -fa -ngl 75 -n 1
		grep prompt_token detlog/* | cut -c17- | sed 's/,//g;s,],,g' >> activs.txt
	done < "$tf"

else
	echo "\nSkipped cpp part 02\n"
fi

if $dotrain02; then
	python ladder_train.py --out_log "res_log_ladder_activations_training_part_02.txt"
else
	echo "\nSkipped train part 02\n"
fi

if $dotest02; then
	python ladder_test.py --out_log "res_log_ladder_activations_testing_part_02.txt"
else
	echo "\nSkipped test part 02\n"
fi

#
### PART 03
#

echo "\n\n#\n###\n#\n################### PART 03 ###################\n#\n###\n#\n\n"

if $docpp03; then

	tf="/home/data/fresh_news_part_02.txt"

	rm -f activs.txt activs.bin
	touch activs.txt

	while IFS="" read -r p || [ -n "$p" ]
	do
        	rm -rf detlog
	        mkdir detlog
        	./llama-cli --logdir detlog --temp 0 -c 2048 -nkvo -m "$modnom" -p "$p" -fa -ngl 75 -n 1
	        grep prompt_token detlog/* | cut -c17- | sed 's/,//g;s,],,g' >> activs.txt
	done < "$tf"

else
        echo "\nSkipped cpp part 03\n"
fi

if $dotrain03; then
        python ladder_train.py --out_log "res_log_ladder_activations_training_part_03.txt"
else
        echo "\nSkipped train part 03\n"
fi

if $dotest03; then
        python ladder_test.py --out_log "res_log_ladder_activations_testing_part_03.txt"
else
        echo "\nSkipped test part 03\n"
fi

#
### PART 04
#

echo "\n\n#\n###\n#\n################### PART 04 ###################\n#\n###\n#\n\n"

if $docpp04; then

	tf="/home/data/fresh_news_part_03.txt"

	rm -f activs.txt activs.bin
	touch activs.txt

	while IFS="" read -r p || [ -n "$p" ]
	do
        	rm -rf detlog
	        mkdir detlog
	        ./llama-cli --logdir detlog --temp 0 -c 2048 -nkvo -m "$modnom" -p "$p" -fa -ngl 75 -n 1
	        grep prompt_token detlog/* | cut -c17- | sed 's/,//g;s,],,g' >> activs.txt
	done < "$tf"

else
        echo "\nSkipped cpp part 04\n"
fi

if $dotrain04; then
        python ladder_train.py --out_log "res_log_ladder_activations_training_part_04.txt"
else
        echo "\nSkipped train part 04\n"
fi

if $dotest04; then
        python ladder_test.py --out_log "res_log_ladder_activations_testing_part_04.txt"
else
        echo "\nSkipped test part 04\n"
fi

#
### PART 05
#

echo "\n\n#\n###\n#\n################### PART 05 ###################\n#\n###\n#\n\n"

if $docpp05; then

	tf="/home/data/fresh_news_part_04.txt"

	rm -f activs.txt activs.bin
	touch activs.txt

	while IFS="" read -r p || [ -n "$p" ]
	do
	        rm -rf detlog
	        mkdir detlog
	        ./llama-cli --logdir detlog --temp 0 -c 2048 -nkvo -m "$modnom" -p "$p" -fa -ngl 75 -n 1
	        grep prompt_token detlog/* | cut -c17- | sed 's/,//g;s,],,g' >> activs.txt
	done < "$tf"

else
        echo "\nSkipped cpp part 05\n"
fi

if $dotrain05; then
        python ladder_train.py --out_log "res_log_ladder_activations_training_part_05.txt"
else
        echo "\nSkipped train part 05\n"
fi

if $dotest05; then
        python ladder_test.py --out_log "res_log_ladder_activations_testing_part_05.txt"
else
        echo "\nSkipped test part 05\n"
fi

#
### PART 06
#

echo "\n\n#\n###\n#\n################### PART 06 ###################\n#\n###\n#\n\n"

if $docpp06; then

	tf="/home/data/fresh_news_part_05.txt"

	rm -f activs.txt activs.bin
	touch activs.txt

	while IFS="" read -r p || [ -n "$p" ]
	do
        	rm -rf detlog
	        mkdir detlog
	        ./llama-cli --logdir detlog --temp 0 -c 2048 -nkvo -m "$modnom" -p "$p" -fa -ngl 75 -n 1
	        grep prompt_token detlog/* | cut -c17- | sed 's/,//g;s,],,g' >> activs.txt
	done < "$tf"

else
        echo "\nSkipped cpp part 06\n"
fi

if $dotrain06; then
        python ladder_train.py --out_log "res_log_ladder_activations_training_part_06.txt"
else
        echo "\nSkipped train part 06\n"
fi

if $dotest06; then
        python ladder_test.py --out_log "res_log_ladder_activations_testing_part_06.txt"
else
        echo "\nSkipped test part 06\n"
fi

#
### PART 07
#

echo "\n\n#\n###\n#\n################### PART 07 ###################\n#\n###\n#\n\n"

if $docpp07; then

	tf="/home/data/fresh_news_part_06.txt"

	rm -f activs.txt activs.bin
	touch activs.txt

	while IFS="" read -r p || [ -n "$p" ]
	do
        	rm -rf detlog
	        mkdir detlog
	        ./llama-cli --logdir detlog --temp 0 -c 2048 -nkvo -m "$modnom" -p "$p" -fa -ngl 75 -n 1
	        grep prompt_token detlog/* | cut -c17- | sed 's/,//g;s,],,g' >> activs.txt
	done < "$tf"

else
        echo "\nSkipped cpp part 07\n"
fi

if $dotrain07; then
        python ladder_train.py --out_log "res_log_ladder_activations_training_part_07.txt"
else
        echo "\nSkipped train part 07\n"
fi

if $dotest07; then
        python ladder_test.py --out_log "res_log_ladder_activations_testing_part_07.txt"
else
        echo "\nSkipped test part 07\n"
fi

#
### PART 08
#

echo "\n\n#\n###\n#\n################### PART 08 ###################\n#\n###\n#\n\n"

if $docpp08; then

	tf="/home/data/fresh_news_part_07.txt"

	rm -f activs.txt activs.bin
	touch activs.txt

	while IFS="" read -r p || [ -n "$p" ]
	do
        	rm -rf detlog
	        mkdir detlog
	        ./llama-cli --logdir detlog --temp 0 -c 2048 -nkvo -m "$modnom" -p "$p" -fa -ngl 75 -n 1
	        grep prompt_token detlog/* | cut -c17- | sed 's/,//g;s,],,g' >> activs.txt
	done < "$tf"

else
        echo "\nSkipped cpp part 08\n"
fi

if $dotrain08; then
        python ladder_train.py --out_log "res_log_ladder_activations_training_part_08.txt"
else
        echo "\nSkipped train part 08\n"
fi

if $dotest08; then
        python ladder_test.py --out_log "res_log_ladder_activations_testing_part_08.txt"
else
        echo "\nSkipped test part 08\n"
fi

#
### PART 09
#

echo "\n\n#\n###\n#\n################### PART 09 ###################\n#\n###\n#\n\n"

if $docpp09; then

	tf="/home/data/fresh_news_part_08.txt"

	rm -f activs.txt activs.bin
	touch activs.txt

	while IFS="" read -r p || [ -n "$p" ]
	do
	        rm -rf detlog
        	mkdir detlog
	        ./llama-cli --logdir detlog --temp 0 -c 2048 -nkvo -m "$modnom" -p "$p" -fa -ngl 75 -n 1
	        grep prompt_token detlog/* | cut -c17- | sed 's/,//g;s,],,g' >> activs.txt
	done < "$tf"

else
        echo "\nSkipped cpp part 09\n"
fi

if $dotrain09; then
        python ladder_train.py --out_log "res_log_ladder_activations_training_part_09.txt"
else
        echo "\nSkipped train part 09\n"
fi

if $dotest09; then
        python ladder_test.py --out_log "res_log_ladder_activations_testing_part_09.txt"
else
        echo "\nSkipped test part 09\n"
fi

#
### PART 10
#

echo "\n\n#\n###\n#\n################### PART 10 ###################\n#\n###\n#\n\n"

if $docpp10; then

	tf="/home/data/fresh_news_part_09.txt"

	rm -f activs.txt activs.bin
	touch activs.txt

	while IFS="" read -r p || [ -n "$p" ]
	do
        	rm -rf detlog
	        mkdir detlog
	        ./llama-cli --logdir detlog --temp 0 -c 2048 -nkvo -m "$modnom" -p "$p" -fa -ngl 75 -n 1
	        grep prompt_token detlog/* | cut -c17- | sed 's/,//g;s,],,g' >> activs.txt
	done < "$tf"

else
        echo "\nSkipped cpp part 10\n"
fi

if $dotrain10; then
        python ladder_train.py --out_log "res_log_ladder_activations_training_part_10.txt"
else
        echo "\nSkipped train part 10\n"
fi

if $dotest10; then
        python ladder_test.py --out_log "res_log_ladder_activations_testing_part_10.txt"
else
        echo "\nSkipped test part 10\n"
fi

#
### END ###
#

