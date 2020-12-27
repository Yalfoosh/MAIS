#include <ctype.h>
#include <malloc.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define BUFFER_SIZE 32
#define N_HEADER_LINES 4
#define N_GROUPS 16
#define FILE_NOT_FOUND_ERROR "File %s not found!"
#define HEADER_ERROR "Expected a 4 line pgm header, but reached EOF at line %d"

// region Helper Functions
char* ltrim(char *s) {
    while(isspace(*s)) {
        s++;
    }

    return s;
}

char* rtrim(char *s) {
    char* back = s + strlen(s);
    while(isspace(*--back));

    *(back+1) = '\0';
    return s;
}

char* trim(char *s) {
    return rtrim(ltrim(s));
}

// endregion

// region Image Data
struct ImageData_s {
    uint8_t** data;
    long long width;
    long long height;
} ImageData_default = {NULL, 0, 0};

typedef struct ImageData_s ImageData;

ImageData* read_image_data(const char* file_path) {
    if(file_path == NULL) {
        file_path = "lenna.pgm";
    }

    FILE *fp = fopen(file_path, "r");

    if(fp == NULL) {
        fprintf(stderr, FILE_NOT_FOUND_ERROR, file_path);
        exit(-1);
    }

    char buffer[BUFFER_SIZE];
    char results[N_HEADER_LINES][BUFFER_SIZE];
    char *ptr;

    // Read header
    for(int i = 0; i < N_HEADER_LINES; ++i) {
        if (!fgets(buffer, sizeof(buffer), fp)) {
            fprintf(stderr, HEADER_ERROR, i + 1);
            exit(-1);
        }

        strcpy(results[i], trim(buffer));
    }

    // Create ImageData
    ImageData* to_return = (ImageData*)malloc(sizeof(ImageData));
    to_return->width = strtoll(results[1], &ptr, 10);
    to_return->height = strtoll(results[2], &ptr, 10);

    // Allocate data matrix
    to_return->data = (uint8_t**)malloc(to_return->height * sizeof(uint8_t*));
    for(uint64_t i = 0; i < to_return->height; ++i) {
        to_return->data[i] = (uint8_t*)malloc(to_return->width * sizeof(uint8_t));
    }

    // Copy data into matrix
    uint8_t row_buffer[to_return->width];

    for(uint64_t i = 0; fread(row_buffer, 1, to_return->width, fp); ++i) {
        for(uint64_t j = 0; j < to_return->width; ++j) {
            to_return->data[i][j] = row_buffer[j];
        }
    }

    return to_return;
}

void free_image_data(ImageData* image_data_ptr) {
    for(uint64_t i = 0; i < image_data_ptr->height; ++i) {
        free(image_data_ptr->data[i]);
    }

    free(image_data_ptr->data);
}

// endregion

// region Group Statistics
uint8_t get_group(uint8_t byte) {
    return byte >> 4;
}

uint64_t* get_group_statistics(ImageData image_data) {
    uint64_t* group_statistics = (uint64_t*)malloc(N_GROUPS * sizeof(uint64_t));

    for(uint64_t i = 0; i < N_GROUPS; ++i) {
        group_statistics[i] = 0;
    }

    for(uint64_t i = 0; i < image_data.height; ++i) {
        for(uint64_t j = 0; j < image_data.width; ++j) {
            ++group_statistics[get_group(image_data.data[i][j]) % N_GROUPS];
        }
    }

    return group_statistics;
}

void free_group_statistics(uint64_t* group_statistics) {
    free(group_statistics);
}
// endregion

int main(int argc, char* argv[]) {
    char* file_path = argc < 2 ? NULL : argv[1];

    ImageData* img = read_image_data(file_path);
    uint64_t* group_statistics = get_group_statistics(*img);

    uint64_t n_total = 0;

    for(uint64_t i = 0; i < N_GROUPS; ++i) {
        n_total += group_statistics[i];
    }

    for(uint64_t i = 0; i < N_GROUPS; ++i) {
        printf("%lu %f\n", i, (double)group_statistics[i] / n_total);
    }

    free_image_data(img);
    free_group_statistics(group_statistics);

    return 0;
}

