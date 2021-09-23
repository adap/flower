cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

function(check_files BASE_DIR FILES_TO_CHECK FILE_MD5S MISSING_FILES)
    foreach(FILE_TO_CHECK ${${FILES_TO_CHECK}})
        if(EXISTS "${BASE_DIR}/${FILE_TO_CHECK}")
            list(FIND ${FILES_TO_CHECK} ${FILE_TO_CHECK} MD5_IDX)
            list(GET ${FILE_MD5S} ${MD5_IDX} EXPECTED_FILE_MD5)
            
            file(MD5 "${BASE_DIR}/${FILE_TO_CHECK}" ACTUAL_FILE_MD5)
            
            if(NOT (EXPECTED_FILE_MD5 STREQUAL ACTUAL_FILE_MD5))
                list(APPEND RESULT_FILES ${FILE_TO_CHECK})
            endif()
        else()
            list(APPEND RESULT_FILES ${FILE_TO_CHECK})
        endif()
    endforeach()
    
    set(${MISSING_FILES} ${RESULT_FILES} PARENT_SCOPE)
endfunction()
