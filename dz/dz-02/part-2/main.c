#include <ctype.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define BUFFER_SIZE 32
#define N_HEADER_LINES 4
#define BLOCK_WIDTH 16
#define BLOCK_HEIGHT 16
#define BLOCK_X_PLUS_DIFF 16
#define BLOCK_X_MINUS_DIFF 16
#define BLOCK_Y_PLUS_DIFF 16
#define BLOCK_Y_MINUS_DIFF 16

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

// region Image Data & Block Data
struct ImageData_s {
    uint8_t** data;
    int64_t width;
    int64_t height;
} ImageData_default = {NULL, 0, 0};

typedef struct ImageData_s ImageData;

struct BlockData_s {
    double** data;
    int64_t width;
    int64_t height;
    uint64_t center_x;
    uint64_t center_y;
} BlockData_default = {NULL, 0, 0, 0, 0};

typedef struct BlockData_s BlockData;

ImageData* read_image_data(const char* file_path) {
    // Open file
    FILE *fp = fopen(file_path, "r");

    if(fp == NULL) {
        fprintf(stderr, FILE_NOT_FOUND_ERROR, file_path);
        exit(-1);
    }

    // Define temporary values
    char buffer[BUFFER_SIZE];
    char results[N_HEADER_LINES][BUFFER_SIZE];
    char *t_ptr;

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
    to_return->width = strtoll(results[1], &t_ptr, 10);
    to_return->height = strtoll(results[2], &t_ptr, 10);

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

void free_block_data(BlockData* block_data_ptr) {
    for(uint64_t i = 0; i < block_data_ptr->height; ++i) {
        free(block_data_ptr->data[i]);
    }

    free(block_data_ptr->data);
}
// endregion

// region Vectors
struct Vector2D_i64_s {
    int64_t x;
    int64_t y;
} Vector2D_i64_default = {0, 0};

typedef struct Vector2D_i64_s Vector2D_i64;

void free_vector2d_i64(Vector2D_i64* vector) {
    free(vector);
}
// endregion

// region Block Operations
ImageData* get_block_from_origin(ImageData image, int64_t origin_x, int64_t origin_y) {
    ImageData* to_return = (ImageData*)malloc(sizeof(ImageData));
    to_return->width = BLOCK_WIDTH;
    to_return->height = BLOCK_HEIGHT;

    to_return->data = (uint8_t**)malloc(to_return->height * sizeof(uint8_t*));
    for(uint64_t i = 0; i < to_return->height; ++i) {
        to_return->data[i] = (uint8_t*)malloc(to_return->width * sizeof(uint8_t));

        for(uint64_t j = 0; j < to_return->width; ++j) {
            to_return->data[i][j] = image.data[origin_y + i][origin_x + j];
        }
    }

    return to_return;
}

double get_block_mad(ImageData reference_block, ImageData interesting_block) {
    uint64_t width_max = reference_block.width;
    uint64_t height_max = reference_block.height;

    if(interesting_block.width < width_max) {
        width_max = interesting_block.width;
    }

    if(interesting_block.height < height_max) {
        height_max = interesting_block.height;
    }

    double result = 0;

    for(uint64_t i = 0; i < height_max; ++i) {
        for(uint64_t j = 0; j < width_max; ++j) {
            if(reference_block.data[i][j] > interesting_block.data[i][j]) {
                result += reference_block.data[i][j] - interesting_block.data[i][j];
            } else {
                result += interesting_block.data[i][j] - reference_block.data[i][j];
            }
        }
    }

    double n_elements = (double)width_max * height_max;

    return result / n_elements;
}
BlockData* get_block_difference(ImageData reference_image, ImageData interesting_image, uint64_t origin_block_index) {
    // Initialize reference block
    uint64_t blocks_per_row = reference_image.width / BLOCK_WIDTH;
    int64_t origin_x = origin_block_index % blocks_per_row;
    int64_t origin_y = origin_block_index / blocks_per_row;

    ImageData* reference_block = get_block_from_origin(reference_image, origin_x, origin_y);

    // Calculate block difference properties
    uint64_t left_offset = BLOCK_X_MINUS_DIFF;
    uint64_t right_offset = BLOCK_X_PLUS_DIFF;
    uint64_t up_offset = BLOCK_Y_MINUS_DIFF;
    uint64_t down_offset = BLOCK_Y_PLUS_DIFF;

    if(origin_x < left_offset) {
        left_offset = origin_x;
    }

    if(origin_y < down_offset) {
        up_offset = origin_y;
    }

    if(origin_x + right_offset >= reference_image.width) {
        right_offset = reference_image.width - origin_x - 1;
    }

    if(origin_y + down_offset >= reference_image.height) {
        down_offset = reference_image.height - origin_y - 1;
    }

    uint64_t x_start = origin_x - left_offset;
    uint64_t y_start = origin_y - up_offset;

    // Initializing block difference
    BlockData* to_return = (BlockData*)malloc(sizeof(BlockData));
    to_return->height = left_offset + right_offset + 1;
    to_return->width = down_offset + up_offset + 1;
    to_return->center_x = left_offset;
    to_return->center_y = up_offset;

    // Allocate data matrix
    to_return->data = (double**)malloc(to_return->height * sizeof(double*));
    for(uint64_t i = 0; i < to_return->height; ++i) {
        to_return->data[i] = (double*)malloc(to_return->width * sizeof(double));
    }

    for(uint64_t i = 0; i < to_return->height; ++i) {
        for(uint64_t j = 0; j < to_return->width; ++j) {
            int64_t current_origin_x = x_start + j;
            int64_t current_origin_y = y_start + i;

            ImageData* t_block = get_block_from_origin(interesting_image, current_origin_x, current_origin_y);

            to_return->data[i][j] = get_block_mad(*reference_block, *t_block);

            free_image_data(t_block);
        }
    }

    free_image_data(reference_block);

    return to_return;
}

// endregion

// region Vector Operations
Vector2D_i64* get_movement_vector(BlockData block_difference) {
    Vector2D_i64* best = (Vector2D_i64*)malloc(sizeof(Vector2D_i64));
    best->x = 0;
    best->y = 0;

    for(uint64_t i = 0; i < block_difference.height; ++i) {
        for(uint64_t j = 0; j < block_difference.width; ++j) {
            if(block_difference.data[i][j] < block_difference.data[best->y][best->x]) {
                best->x = j;
                best->y = i;
            }
        }
    }

    best->x -= block_difference.center_x;
    best->y -= block_difference.center_y;

    return best;
}

// endregion

int main(int argc, char* argv[]) {
    uint64_t block_index = 0;
    char* reference_image_path = NULL;
    char* interesting_image_path = NULL;

    char* t_ptr;

    if(argc > 1) {
        block_index = strtoll(argv[1], &t_ptr, 10);

        if(argc > 2) {
            reference_image_path = argv[2];

            if(argc > 3) {
                interesting_image_path = argv[3];
            }
        }
    }

    if(reference_image_path == NULL) {
        reference_image_path = "lenna.pgm";
    }

    if(interesting_image_path == NULL) {
        interesting_image_path = "lenna1.pgm";
    }

    ImageData* reference_img = read_image_data(reference_image_path);
    ImageData* interesting_img = read_image_data(interesting_image_path);

    BlockData* block_difference = get_block_difference(*reference_img, *interesting_img, block_index);
    Vector2D_i64* movement_vector = get_movement_vector(*block_difference);

    printf("(%ld,%ld)\n", movement_vector->x, movement_vector->y);

    free_vector2d_i64(movement_vector);
    free_block_data(block_difference);
    free_image_data(interesting_img);
    free_image_data(reference_img);

    return 0;
}
