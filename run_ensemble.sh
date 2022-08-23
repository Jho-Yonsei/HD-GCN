printf "\nNTU-RGB+D 60 Cross-Subject\n"
python ensemble.py \
	--dataset ntu/xsub \
	--main-dir ./work_dir/ntu/cross-subject/

printf "\nNTU-RGB+D 60 Cross-View\n"
python ensemble.py \
	--dataset ntu/xview \
	--main-dir ./work_dir/ntu/cross-view/

printf "\nNTU-RGB+D 120 Cross-Subject\n"
python ensemble.py \
	--dataset ntu120/xsub \
	--main-dir ./work_dir/ntu120/cross-subject/

printf "\nNTU-RGB+D 120 Cross-Setup\n"
python ensemble.py \
	--dataset ntu120/xset \
	--main-dir ./work_dir/ntu120/cross-setup/

printf "\n"
