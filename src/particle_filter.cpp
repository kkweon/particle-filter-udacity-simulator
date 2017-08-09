/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <math.h>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <iostream>
#include <iterator>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <string>

#include "particle_filter.h"

using namespace std;

namespace {
// file scope helper functions and variables are defined here
std::default_random_engine gen;

template <typename T>
constexpr T square(T val) {
  return val * val;
}

// get distance between two landmarks
double get_dist(const LandmarkObs& landmark1, const LandmarkObs& landmark2) {
  auto x_diff = landmark1.x - landmark2.x;
  auto y_diff = landmark1.y - landmark2.y;

  return sqrt(pow(x_diff, 2) + pow(y_diff, 2));
}

double get_dist(const Particle& landmark1,
                const Map::single_landmark_s& landmark2) {
  auto x_diff = landmark1.x - landmark2.x_f;
  auto y_diff = landmark1.y - landmark2.y_f;

  return sqrt(pow(x_diff, 2) + pow(y_diff, 2));
}

LandmarkObs rotate(const LandmarkObs& obs, const Particle& particle) {
  LandmarkObs rotated_obs;
  float theta = particle.theta;
  rotated_obs.id = obs.id;
  rotated_obs.x = particle.x + obs.x * cos(theta) - obs.y * sin(theta);
  rotated_obs.y = particle.y + obs.x * sin(theta) + obs.y * cos(theta);

  return rotated_obs;
}
/**
 * Compute Gaussian2D (Multivariate Gaussian)
 *
 * @param obs Landmark Observation
 * @param landmark Actual Landmark Predicted by Nearest Neighbor
 * @param sigma Standard Deviation of x, y
 *
 * @return Gaussian probability
 */
double gaussian2d(const LandmarkObs& obs, const LandmarkObs& landmark,
                  const double sigma[]) {
  auto sigma_x = sigma[0];
  auto sigma_y = sigma[1];

  auto sigma_x_sq = square(sigma_x);
  auto sigma_y_sq = square(sigma_y);

  auto x_diff_sq = square(obs.x - landmark.x);
  auto y_diff_sq = square(obs.y - landmark.y);

  auto normalizer = 2 * M_PI * sigma_x * sigma_y;

  return exp(-x_diff_sq / (2 * sigma_x_sq) - y_diff_sq / (2 * sigma_y_sq)) /
         normalizer;
}

}  // namespace

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  // TODO: Set the number of particles. Initialize all particles to first
  // position (based on estimates of
  //   x, y, theta and their uncertainties from GPS) and all weights to 1.
  // Add random Gaussian noise to each particle.
  // NOTE: Consult particle_filter.h for more information about this method (and
  // others in this file).
  num_particles = 100;

  std::normal_distribution<double> get_normal_x(x, std[0]);
  std::normal_distribution<double> get_normal_y(y, std[1]);
  std::normal_distribution<double> get_normal_theta(theta, std[2]);

  weights.resize(num_particles, 1.0f);
  particles.resize(num_particles);
  for (int i = 0; i < num_particles; ++i) {
    Particle new_particle;
    new_particle.id = i;
    new_particle.x = get_normal_x(gen);
    new_particle.y = get_normal_y(gen);
    new_particle.theta = get_normal_theta(gen);
    new_particle.weight = 1;
    particles[i] = new_particle;
  }
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate) {
  // NOTE: If we know velocity and yaw_rate and previous (x, y)
  // Then we can predict next (x, y) coordinates.
  // Also add some gaussian noises with given standard deviations

  std::normal_distribution<double> get_noise_x(0, std_pos[0]);
  std::normal_distribution<double> get_noise_y(0, std_pos[1]);
  std::normal_distribution<double> get_noise_theta(0, std_pos[2]);

  for (auto& particle : particles) {
    if (fabs(yaw_rate) < 1e-5) {
      // if yaw_rate is near zero, the formula is different
      particle.x += velocity * delta_t * cos(particle.theta);
      particle.y += velocity * delta_t * sin(particle.theta);
      // particle.theta does not change since yaw rate is 0
    } else {
      auto theta_change = particle.theta + yaw_rate * delta_t;
      particle.x +=
          velocity / yaw_rate * (sin(theta_change) - sin(particle.theta));
      particle.y +=
          velocity / yaw_rate * (cos(particle.theta) - cos(theta_change));
      particle.theta = theta_change;
    }

    particle.x += get_noise_x(gen);
    particle.y += get_noise_y(gen);
    particle.theta += get_noise_theta(gen);
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted,
                                     std::vector<LandmarkObs>& observations) {
  // TODO: Find the predicted measurement that is closest to each observed
  // measurement and assign the
  //   observed measurement to this particular landmark.
  // NOTE: this method will NOT be called by the grading code. But you will
  // probably find it useful to
  //   implement this method and use it as a helper during the updateWeights
  //   phase.

  for (auto& obs_landmark : observations) {
    int min_index = -1;
    double min_distance = numeric_limits<double>::infinity();

    for (auto i = 0; i < predicted.size(); ++i) {
      auto landmark = predicted[i];
      auto dist = get_dist(obs_landmark, landmark);

      if (dist < min_distance) {
        min_distance = dist;
        min_index = i;  // landmark.id;
      }
    }
    assert(min_index >= 0);
    obs_landmark.id = min_index;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   std::vector<LandmarkObs> observations,
                                   Map map_landmarks) {
  // TODO: Update the weights of each particle using a mult-variate Gaussian
  // distribution. You can read
  //   more about this distribution here:
  //   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
  // NOTE: The observations are given in the VEHICLE'S coordinate system. Your
  // particles are located
  //   according to the MAP'S coordinate system. You will need to transform
  //   between the two systems.
  //   Keep in mind that this transformation requires both rotation AND
  //   translation (but no scaling).
  //   The following is a good resource for the theory:
  //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
  //   and the following is a good resource for the actual equation to
  //   implement
  //   (look at equation
  //   3.33
  //   http://planning.cs.uiuc.edu/node99.html

  for (auto i = 0; i < particles.size(); ++i) {
    auto particle = particles[i];

    // Step 1. Get landmarks within the sensor range
    // Call it predicted landmarks
    std::vector<LandmarkObs> predicted_landmarks;
    for (auto& landmark : map_landmarks.landmark_list) {
      auto dist = get_dist(particle, landmark);
      if (dist <= sensor_range) {
        LandmarkObs landmark_predicted;
        landmark_predicted.id = landmark.id_i;
        landmark_predicted.x = landmark.x_f;
        landmark_predicted.y = landmark.y_f;

        predicted_landmarks.push_back(landmark_predicted);
      }
    }

    // Step 2. Observations are in the perspective of the vehicle.
    // Rotate the coordinates to the global map coordinates
    std::vector<LandmarkObs> transformed_observations;
    double total_prob = 1.0f;

    for (auto& obs : observations) {
      transformed_observations.push_back(rotate(obs, particle));
    }

    // Step 3. Each observation must be associated with predicted landmarks
    dataAssociation(predicted_landmarks, transformed_observations);

    // Step 4. Check how likely the observation is.
    // Gaussian 2D distribution from landmark location
    for (auto& obs : transformed_observations) {
      auto landmark = predicted_landmarks[obs.id];
      total_prob *= gaussian2d(obs, landmark, std_landmark);
    }

    particles[i].weight = total_prob;
    weights[i] = total_prob;
  }
}

void ParticleFilter::resample() {
  // TODO: Resample particles with replacement with probability proportional
  // to
  // their weight.
  // NOTE: You may find std::discrete_distribution helpful here.
  //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
  std::discrete_distribution<int> discrete_dist(std::begin(weights),
                                                std::end(weights));
  std::vector<Particle> resampled_particles(weights.size());

  for (auto i = 0; i < weights.size(); ++i) {
    resampled_particles[i] = particles[discrete_dist(gen)];
  }
  particles = resampled_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle,
                                         std::vector<int> associations,
                                         std::vector<double> sense_x,
                                         std::vector<double> sense_y) {
  // particle: the particle to assign each listed association, and
  // association's
  // (x,y) world coordinates mapping to
  // associations: The landmark id that goes along with each listed
  // association
  // sense_x: the associations x mapping already converted to world
  // coordinates
  // sense_y: the associations y mapping already converted to world
  // coordinates

  // Clear the previous associations
  particle.associations.clear();
  particle.sense_x.clear();
  particle.sense_y.clear();

  particle.associations = associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;

  return particle;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}
string ParticleFilter::getSenseX(Particle best) {
  vector<double> v = best.sense_x;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}
string ParticleFilter::getSenseY(Particle best) {
  vector<double> v = best.sense_y;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}
