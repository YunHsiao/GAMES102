#pragma once

#include <UGM/UGM.h>
#include <Eigen/Core>

enum LossFunction {
	MeanSquaredError,
	CrossEntropy,
};

struct CanvasData {
	std::vector<Ubpa::pointf2> input_points;
	std::vector<Ubpa::pointf2> point_set;
	Ubpa::valf2 scrolling{ 0.f,0.f };
	bool opt_enable_grid{ true };
	bool opt_enable_context_menu{ true };

	int opt_fitting_step{ 5 };
	bool opt_is_training{ false };
	int opt_hidden_node_count{ 10 };
	float opt_learning_rate{ 0.001f };
	int opt_loss_function{ LossFunction::MeanSquaredError };

	Eigen::MatrixXf ONES;
	Eigen::MatrixXf X;
	Eigen::MatrixXf Y;

	Eigen::MatrixXf W;
	Eigen::MatrixXf b;
	Eigen::MatrixXf W2;
	float b2{ 0.f };

	Eigen::MatrixXf dW;
	Eigen::MatrixXf db;
	Eigen::MatrixXf dW2;
	float db2{ 0.f };

	Eigen::MatrixXf Z;
	Eigen::MatrixXf A;
	Eigen::MatrixXf Y_predict;

	float cost{ 0.f };
	size_t iteration{ 0 };
	size_t current_count{ 0u };

};

#include "details/CanvasData_AutoRefl.inl"
