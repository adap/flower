import { FlowerIntelligence } from '@flwr/flwr';

const fi = FlowerIntelligence.instance;
fi.remoteHandoff = true;

fi.apiKey = process.env.FI_API_KEY;

const numTries = 300;
let successCount = 0;
const errorLogs = [];

const testEmail = `
Hi Johnson,

I hope you're well.

We're making good progress on the model side, but it's not that "visible" / there's not much to sync on. I could give a quick update on our progress with the UI, but we also want to be respectful of your time and offer to cancel.

We'll be at a company-wide Get Together in France next week, so we'll have to cancel our sync next week. If we also cancel tomorrow, we'd have the next meeting in two weeks.

Let us know what you prefer.

Jack
`;

console.log('Starting stress test...');
const requestTimes = [];
const responses = [];
const overallStart = Date.now();
const requests = Array.from({ length: numTries }, (_, i) => {
  const startTime = Date.now();

  return fi
    .chat({
      messages: [
        {
          role: 'user',
          content: `Can you write a short summary the following email (your reply should only contain the short summary)?\n${testEmail}`,
        },
      ],
      forceRemote: true,
    })
    .then((response) => {
      const endTime = Date.now();
      requestTimes.push({ index: i, time: endTime - startTime }); // Calculate and store duration

      if (!response.ok) {
        process.stdout.write('-');
        errorLogs.push(`Error ${i}: ${response.failure.description}`);
      } else {
        successCount++;
        process.stdout.write('+');
        responses.push(response.message.content);
      }
    })
    .catch((e) => {
      const endTime = Date.now(); // Record the end time in case of error
      requestTimes.push({ index: i, time: endTime - startTime }); // Calculate and store duration

      process.stdout.write('-');
      errorLogs.push(`Error ${i}: ${e.message}`);
    });
});

// Wait for all requests to complete
Promise.allSettled(requests).then(() => {
  const overallEnd = Date.now();
  if (successCount !== numTries) {
    if (errorLogs.length > 0) {
      console.log('\nError logs:');
      errorLogs.forEach((log) => console.log(log));
    }
    console.error(`Test failed: ${successCount}/${numTries} requests succeeded.`);
  } else {
    console.log(`\nTest passed: All ${successCount} requests succeeded.`);
  }

  // Output request timings
  if (requestTimes.length > 0) {
    const best = requestTimes.reduce((prev, curr) => (prev.time < curr.time ? prev : curr)); // Fastest request
    const worst = requestTimes.reduce((prev, curr) => (prev.time > curr.time ? prev : curr)); // Slowest request
    const average = requestTimes.reduce((sum, { time }) => sum + time, 0) / requestTimes.length; // Average request time
    const median = calculateMedianWithIndex(requestTimes); // Median request time

    console.log('\nSummary of Request Times:');
    console.log(`Overall time: ${overallEnd - overallStart}ms`);
    console.log(`Best (Fastest): Index ${best.index}, Time ${best.time}ms`);
    console.log(`Worst (Slowest): Index ${worst.index}, Time ${worst.time}ms`);
    console.log(`Median: Time ${median.time.toFixed(2)}ms, Indices ${median.indices.join(', ')}`);
    console.log(`Average: ${average.toFixed(2)}ms`);
  }
  const randomIndex = Math.floor(Math.random() * responses.length);
  console.log(`Sampled response:\n${responses[randomIndex]}`);
});

function calculateMedianWithIndex(arr) {
  const sorted = arr
    .map((item, index) => ({ ...item, originalIndex: index })) // Preserve original index
    .sort((a, b) => a.time - b.time); // Sort by time in ascending order

  const mid = Math.floor(sorted.length / 2);
  if (sorted.length % 2 === 0) {
    // Average of two middle elements
    const left = sorted[mid - 1];
    const right = sorted[mid];
    return {
      time: (left.time + right.time) / 2,
      indices: [left.originalIndex, right.originalIndex], // Include both indices
    };
  } else {
    // Single middle element
    return {
      time: sorted[mid].time,
      indices: [sorted[mid].originalIndex],
    };
  }
}
