enum NovaMode {
  navigation,
  reading,
  recognition,
  emergency,
}

extension NovaModeName on NovaMode {
  String get label {
    switch (this) {
      case NovaMode.navigation:
        return "Navigation mode";
      case NovaMode.reading:
        return "Reading mode";
      case NovaMode.recognition:
        return "Recognition mode";
      case NovaMode.emergency:
        return "Emergency mode";
    }
  }
}
