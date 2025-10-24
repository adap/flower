package ai.flower.intelligence.examples.helloworld

import ai.flower.intelligence.FlowerIntelligence
import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.navigation.NavHostController
import androidx.navigation.compose.*
import kotlinx.coroutines.launch

class MainActivity : ComponentActivity() {
  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    Log.d("API_KEY", "Using key: ${BuildConfig.API_KEY}")
    FlowerIntelligence.apiKey = BuildConfig.API_KEY

    setContent { MaterialTheme { AppNavigator() } }
  }
}

@Composable
fun AppNavigator() {
  val navController = rememberNavController()
  NavHost(navController = navController, startDestination = "flower") {
    composable("home") { HomeScreen(navController) }
    composable("flower") { FlowerApp() }
  }
}

@Composable
fun HomeScreen(navController: NavHostController) {
  Surface(modifier = Modifier.fillMaxSize()) {
    Column(
      modifier = Modifier.padding(24.dp).fillMaxSize(),
      verticalArrangement = Arrangement.Center,
      horizontalAlignment = Alignment.CenterHorizontally,
    ) {
      Text("Welcome to Flower Intelligence!")
      Spacer(modifier = Modifier.height(16.dp))
      Button(onClick = { navController.navigate("flower") }) { Text("Ask a Question") }
    }
  }
}

@Composable
fun FlowerApp() {
  val scope = rememberCoroutineScope()
  var input by remember { mutableStateOf("") }
  var response by remember { mutableStateOf("") }
  var loading by remember { mutableStateOf(false) }

  Surface(modifier = Modifier.fillMaxSize()) {
    Column(
      modifier = Modifier.padding(16.dp).fillMaxWidth(),
      verticalArrangement = Arrangement.Center,
      horizontalAlignment = Alignment.CenterHorizontally,
    ) {
      if (loading) {
        CircularProgressIndicator()
      } else if (response.isNotBlank()) {
        Column(
          modifier =
            Modifier.fillMaxWidth().weight(1f).verticalScroll(rememberScrollState()).padding(8.dp)
        ) {
          Text(response)
        }
      }

      Spacer(modifier = Modifier.height(24.dp))

      OutlinedTextField(
        value = input,
        onValueChange = { input = it },
        label = { Text("Type a message...") },
        modifier = Modifier.fillMaxWidth(),
      )

      Spacer(modifier = Modifier.height(16.dp))

      Button(
        onClick = {
          loading = true
          response = ""
          scope.launch {
            val result = FlowerIntelligence.chat(input)
            val message = result.getOrNull()
            response = message?.content ?: result.exceptionOrNull()?.message ?: "Unknown error"
            loading = false
          }
        },
        enabled = input.isNotBlank() && !loading,
      ) {
        Text("Send")
      }
    }
  }
}
