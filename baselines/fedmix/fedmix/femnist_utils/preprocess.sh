#!/usr/bin/env bash
#
# script to preprocess data

# --------------------
# parse arguments

NAME="sent140" # name of the dataset, equivalent to directory name
SAMPLE="na" # -s tag, iid or niid
IUSER="" # --iu tag, # of users if iid sampling
SFRAC="" # --sf tag, fraction of data to sample
MINSAMPLES="na" # -k tag, minimum allowable # of samples per user
TRAIN="na" # -t tag, user or sample
TFRAC="" # --tf tag, fraction of data in training set
SAMPLING_SEED="" # --smplseed, seed specified for sampling of data
SPLIT_SEED="" # --spltseed, seed specified for train/test data split
NO_CHECKSUM="" # --nochecksum, disable creation of MD5 checksum file after data gen
VERIFICATION_FILE="" # --verify <fname>, check if JSON files' MD5 matches given digest

META_DIR='meta'
CHECKSUM_FNAME="${META_DIR}/dir-checksum.md5"

while [[ $# -gt 0 ]]
do
key="$1"

# TODO: Use getopts instead of creating cases!
case $key in
    --name)
    NAME="$2"
    shift # past argument
    if [ ${SAMPLE:0:1} = "-" ]; then
        NAME="sent140"
    else
        shift # past value
    fi
    ;;
    -s)
    SAMPLE="$2"
    shift # past argument
    if [ ${SAMPLE:0:1} = "-" ]; then
        SAMPLE=""
    else
        shift # past value
    fi
    ;;
    --iu)
    IUSER="$2"
    shift # past argument
    if [ ${IUSER:0:1} = "-" ]; then
        IUSER=""
    else
        shift # past value
    fi
    ;;
    --sf)
    SFRAC="$2"
    shift # past argument
    if [ ${SFRAC:0:1} = "-" ]; then
        SFRAC=""
    else
        shift # past value
    fi
    ;;
    -k)
    MINSAMPLES="$2"
    shift # past argument
    if [ ${MINSAMPLES:0:1} = "-" ]; then
        MINSAMPLES=""
    else
        shift # past value
    fi
    ;;
    -t)
    TRAIN="$2"
    shift # past argument
    if [ -z "$TRAIN" ] || [ ${TRAIN:0:1} = "-" ]; then
        TRAIN=""
    else
        shift # past value
    fi
    ;;
    --tf)
    TFRAC="$2"
    shift # past argument
    if [ ${TFRAC:0:1} = "-" ]; then
        TFRAC=""
    else
        shift # past value
    fi
    ;;
    --smplseed)
    SAMPLING_SEED="$2"
    shift # past argument
    ;;
    --spltseed)
    SPLIT_SEED="$2"
    shift # past argument
    ;;
    --nochecksum)
    NO_CHECKSUM="true"
    shift # past argument
    ;;
    --verify)
    VERIFICATION_FILE="$2"
    shift # past argument
    ;;
    *)    # unknown option
    shift # past argument
    ;;
esac
done

# --------------------
# check if running in verification mode

if [ -n "${VERIFICATION_FILE}" ]; then
    pushd ../$NAME >/dev/null 2>/dev/null
        TMP_FILE=`mktemp /tmp/dir-checksum.XXXXXXX`
        find 'data/' -type f -name '*.json' -exec md5sum {} + | sort -k 2 > ${TMP_FILE}
        DIFF=`diff --brief ${VERIFICATION_FILE} ${TMP_FILE}`
        if [ $? -ne 0 ]; then
            echo "${DIFF}"
            diff ${TMP_FILE} ${VERIFICATION_FILE} 
            echo "Differing checksums found - please verify"
        else
            echo "Matching JSON files and checksums found!"
        fi
    popd >/dev/null 2>/dev/null
    exit 0
fi

# --------------------
# preprocess data

CONT_SCRIPT=true
cd ../$NAME

# setup meta directory if doesn't exist
if [ ! -d ${META_DIR} ]; then
    mkdir -p ${META_DIR}
fi
META_DIR=`realpath ${META_DIR}`

# download data and convert to .json format

if [ ! -d "data/all_data" ]; then
    cd preprocess
    ./data_to_json.sh
    cd ..
fi

NAMETAG="--name $NAME"

# sample data
IUSERTAG=""
if [ ! -z $IUSER ]; then
    IUSERTAG="--u $IUSER"
fi
SFRACTAG=""
if [ ! -z $SFRAC ]; then
    SFRACTAG="--fraction $SFRAC"
fi

if [ "$CONT_SCRIPT" = true ] && [ ! $SAMPLE = "na" ]; then
    if [ -d "data/sampled_data" ] && [ "$(ls -A data/sampled_data)" ]; then
        CONT_SCRIPT=false
    else
        if [ ! -d "data/sampled_data" ]; then
            mkdir data/sampled_data
        fi

        cd ../femnist_utils

        # Defaults to -1 if not specified, causes script to randomly generate seed
        SEED_ARGUMENT="${SAMPLING_SEED:--1}" 

        if [ $SAMPLE = "iid" ]; then
            LEAF_DATA_META_DIR=${META_DIR} python3 sample.py $NAMETAG --iid $IUSERTAG $SFRACTAG --seed ${SEED_ARGUMENT}
        else
            LEAF_DATA_META_DIR=${META_DIR} python3 sample.py $NAMETAG $SFRACTAG --seed ${SEED_ARGUMENT}
        fi

        cd ../$NAME
    fi
fi

# remove users with less then given number of samples
if [ "$CONT_SCRIPT" = true ] && [ ! $MINSAMPLES = "na" ]; then
    if [ -d "data/rem_user_data" ] && [ "$(ls -A data/rem_user_data)" ]; then
        CONT_SCRIPT=false
    else
        if [ ! -d "data/rem_user_data" ]; then
            mkdir data/rem_user_data
        fi

        cd ../femnist_utils

        if [ -z $MINSAMPLES ]; then
            python3 remove_users.py $NAMETAG
        else
            python3 remove_users.py $NAMETAG --min_samples $MINSAMPLES
        fi

        cd ../$NAME
    fi
fi

# create train-test split
TFRACTAG=""
if [ ! -z $TFRAC ]; then
    TFRACTAG="--frac $TFRAC"
fi

if [ "$CONT_SCRIPT" = true ] && [ ! $TRAIN = "na" ]; then
    if [ -d "data/train" ] && [ "$(ls -A data/train)" ]; then
        CONT_SCRIPT=false
    else
        if [ ! -d "data/train" ]; then
            mkdir data/train
        fi
        if [ ! -d "data/test" ]; then
            mkdir data/test
        fi

        cd ../femnist_utils

        # Defaults to -1 if not specified, causes script to randomly generate seed
        SEED_ARGUMENT="${SPLIT_SEED:--1}"

        if [ -z $TRAIN ]; then
            LEAF_DATA_META_DIR=${META_DIR} python3 split_data.py $NAMETAG $TFRACTAG --seed ${SEED_ARGUMENT}
        elif [ $TRAIN = "user" ]; then
            LEAF_DATA_META_DIR=${META_DIR} python3 split_data.py $NAMETAG --by_user $TFRACTAG --seed ${SEED_ARGUMENT}
        elif [ $TRAIN = "sample" ]; then
            LEAF_DATA_META_DIR=${META_DIR} python3 split_data.py $NAMETAG --by_sample $TFRACTAG --seed ${SEED_ARGUMENT}
        fi

        cd ../$NAME
    fi
fi

if [ -z "${NO_CHECKSUM}" ]; then
    echo '------------------------------'
    echo "calculating JSON file checksums"
    find 'data/' -type f -name '*.json' -exec md5sum {} + | sort -k 2 > ${CHECKSUM_FNAME}
    echo "checksums written to ${CHECKSUM_FNAME}"
fi

if [ "$CONT_SCRIPT" = false ]; then
    echo "Data for one of the specified preprocessing tasks has already been"
    echo "generated. If you would like to re-generate data for this directory,"
    echo "please delete the existing one. Otherwise, please remove the"
    echo "respective tag(s) from the preprocessing command."
fi
