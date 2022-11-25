# GEOLIFE AUTO LABELING
python3 ProposedApproach/AutomaticLabeling.py --dataset geolife

# CYPRUS AUTO LABELING
python3 ProposedApproach/AutomaticLabeling.py --dataset geolife --G 10

# GEOLIFE CLASSIFICATION
python3 TOP/runTOP.py --dataset geolife
python3 DODB/DODB.py --dataset geolife
python3 ProposedApproach/Proposed.py --dataset geolife --method clustering --minPts 20 --distance_fn hausdorff --G 20
python3 ProposedApproach/Proposed.py --dataset geolife --method clustering --minPts 2 --distance_fn dtw --G 20
python3 ProposedApproach/Proposed.py --dataset geolife --method clustering --minPts 2 --distance_fn dtw_hilbert --G 20
python3 ProposedApproach/Proposed.py --dataset geolife --method svm --do_gsp 0
python3 ProposedApproach/Proposed.py --dataset geolife --method svm --do_gsp 1 --G 20
python3 ProposedApproach/Proposed.py --dataset geolife --method both --minPts 5 --distance_fn hausdorff --G 20 --do_gsp 1

# CYPRUS CLASSIFICATION
python3 TOP/runTOP.py --dataset cyprus --seqGap 6 --minSup 3
python3 DODB/DODB.py --dataset cyprus -W 2 -D1 30000
python3 ProposedApproach/Proposed.py --dataset cyprus --method clustering --minPts 30 --distance_fn hausdorff --G 40
python3 ProposedApproach/Proposed.py --dataset cyprus --method clustering --minPts 5 --distance_fn dtw --G 40
python3 ProposedApproach/Proposed.py --dataset cyprus --method clustering --minPts 2 --distance_fn dtw_hilbert --G 40
python3 ProposedApproach/Proposed.py --dataset cyprus --method svm --do_gsp 0
python3 ProposedApproach/Proposed.py --dataset cyprus --method svm --do_gsp 1
python3 ProposedApproach/Proposed.py --dataset cyprus --method both --minPts 30 --distance_fn hausdorff --G 40 --do_gsp 1
